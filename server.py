import os
import io
import base64
import logging
import time
import json
from contextlib import asynccontextmanager
from threading import Thread
from typing import Optional, List, Dict, Any, AsyncGenerator
from transformers import BitsAndBytesConfig
import torch
import uvicorn
import GPUtil
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field,field_validator
from PIL import Image, ImageFile
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)

# --- Configuration ---
# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model Configuration (Adjust as needed)
MODEL_PATH = 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'
# Or your local path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
USE_FLASH_ATTENTION_2 = True # Set to True if flash-attn is installed and compatible

# --- Global Variables ---
model = None
processor = None
tokenizer = None # Keep tokenizer separate for streamer

# --- Helper Functions ---

def log_system_info(detail=""):
    """Log system resource information"""
    try:
        prefix = f"System Info ({detail})" if detail else "System Info"
        cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking
        memory = psutil.virtual_memory()
        gpu_info_str = "N/A"
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0] # Assuming single GPU relevant for loading
                gpu_info_str = (f"GPU ID: {gpu.id}, Name: {gpu.name}, "
                                f"Load: {gpu.load*100:.1f}%, "
                                f"Mem: {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f}MB, "
                                f"Temp: {gpu.temperature}°C")
            else:
                gpu_info_str = "CUDA available but GPUtil failed to get details."

        logger.info(f"{prefix} - CPU: {cpu_percent}%, RAM: {memory.percent}% ({memory.available/1024**3:.1f}GB free), {gpu_info_str}")
    except Exception as e:
        logger.warning(f"Failed to log system info: {str(e)}")

def process_base64_image(base64_string: str) -> Image.Image:
    """Process base64 image data and return PIL Image"""
    try:
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        # Convert to RGB for consistency, Qwen-VL might handle others but RGB is safe
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}", exc_info=True)
        raise ValueError(f"Invalid base64 image data provided.") from e

# --- FastAPI Application Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and clear resources on shutdown."""
    global model, processor, tokenizer, USE_FLASH_ATTENTION_2
    logger.info("Starting application initialization...")
    log_system_info("Startup Begin")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, running on CPU. This will be very slow.")
        USE_FLASH_ATTENTION_2 = False # Flash Attn requires CUDA

    if USE_FLASH_ATTENTION_2:
        try:
            import flash_attn
            logger.info("Flash Attention 2 is available and enabled.")
            attn_implementation = "flash_attention_2"
        except ImportError:
            logger.warning("Flash Attention 2 requested but not found. Falling back to default.")
            USE_FLASH_ATTENTION_2 = False
            attn_implementation = "sdpa" # Or None, depending on transformers version
    else:
         attn_implementation = "sdpa" # Use Scaled Dot Product Attention if available
         logger.info("Using default attention implementation (SDPA if available).")


    try:
        logger.info(f"Loading model '{MODEL_PATH}' onto {DEVICE} with dtype {TORCH_DTYPE}...")
        start_time = time.time()

        # Dynamically get the class to handle potential custom code in the model repo
        # model_cls = get_class_from_dynamic_module(MODEL_PATH, "modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration")
        # processor_cls = get_class_from_dynamic_module(MODEL_PATH, "processing_qwen2_5_vl.Qwen2_5VLProcessor")


        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
             torch_dtype=torch.float16,
            device_map="auto", # Simple mapping for single GPU or CPU
            attn_implementation=attn_implementation if USE_FLASH_ATTENTION_2 else "sdpa", # Use FA2 if enabled
        ).eval()

        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        tokenizer = processor.tokenizer # Store tokenizer separately for the streamer

        end_time = time.time()
        logger.info(f"Model and processor loaded successfully in {end_time - start_time:.2f} seconds.")
        log_system_info("Model Loaded")

    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}", exc_info=True)
        # Optionally raise to prevent server start, or handle gracefully
        model = None
        processor = None
        tokenizer = None
        # raise RuntimeError("Failed to initialize model") from e # Uncomment to force stop on load failure

    yield # Application runs here

    # Cleanup on shutdown
    logger.info("Shutting down application...")
    model = None
    processor = None
    tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared.")
    log_system_info("Shutdown Complete")

app = FastAPI(
    title="Qwen2.5-VL Video Description API",
    description="API for generating descriptions from video frames using Qwen2.5-VL with streaming.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request/Response ---

class VideoDescriptionRequest(BaseModel):
    images: List[str] = Field(..., min_items=1, max_items=512, description="List of base64 encoded image strings representing video frames.")
    prompt: str = Field(default="Describe in detail what is happening in this sequence of images.", description="Prompt to guide the model's description.")
    max_new_tokens: int = Field(default=1024, gt=0, le=8192, description="Maximum number of new tokens to generate.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling top-p.")
    # stream: bool = Field(default=True, description="Always true for this endpoint") # Implicitly true

    @field_validator('images')
    @classmethod
    def check_images_not_empty(cls, v):
        if not v:
            raise ValueError("Image list cannot be empty.")
        # Basic base64 check (optional, might add latency)
        # for i, img_str in enumerate(v):
        #     if not img_str or not isinstance(img_str, str):
        #          raise ValueError(f"Image at index {i} is not a valid string.")
        return v

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    log_system_info("Health Check")
    model_ready = model is not None and processor is not None
    return JSONResponse(content={
        "status": "healthy" if model_ready else "degraded",
        "model_ready": model_ready,
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "torch_dtype": str(TORCH_DTYPE),
        "using_flash_attention_2": USE_FLASH_ATTENTION_2,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": time.time()
    })
@app.post("/v1/video/describe")
async def describe_video(request: VideoDescriptionRequest):
    if model is None or processor is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not available or not loaded properly.")

    request_start_time = time.time()
    logger.info(f"Received video description request with {len(request.images)} frames.")
    log_system_info("Request Start")

    try:
        # 1. Process Base64 Images into PIL Images
        pil_images: List[Image.Image] = []
        for i, img_str in enumerate(request.images):
            try:
                pil_images.append(process_base64_image(img_str))
            except ValueError as e:
                logger.error(f"Failed to decode base64 image at index {i}: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid base64 image data at index {i}.")
            except Exception as e:
                 logger.error(f"Unexpected error processing image {i}: {e}", exc_info=True)
                 raise HTTPException(status_code=500, detail=f"Error processing image at index {i}.")

        logger.info(f"Successfully decoded {len(pil_images)} images into PIL format.")

        # 2. Prepare input using the Multi-Image approach
        # Construct the message structure for multi-image input
        user_content = []
        # Add image placeholders (the processor matches these with the `images` argument)
        for _ in range(len(pil_images)):
            user_content.append({"type": "image"})
        # Add the text prompt
        user_content.append({"type": "text", "text": request.prompt})

        messages = [{"role": "user", "content": user_content}]
        # --- Step 2a: Apply chat template to get the formatted text string ---
        # This converts the structured message list into a single string with
        # appropriate chat role markers and image placeholders/tokens.
        try:
            text_prompt_string = processor.apply_chat_template(
                messages,
                tokenize=False, # Get the string representation
                add_generation_prompt=True # Add the prompt marker for generation
            )
            logger.info("Applied chat template to format text string.") 
            logger.debug(f"Formatted text prompt string: {text_prompt_string[:200]}...") # Log start of string
        except Exception as e:
            logger.error(f"Error during apply_chat_template: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error formatting prompt: {e}")
        # --- Step 2b: Call the main processor with the text string and PIL images ---
        # Now, the processor takes the formatted string and the actual images.
        # It will handle tokenizing the text and processing the images.
        try:
            # This is the standard way shown in docs for multi-image
            inputs = processor(
                text=[text_prompt_string],      # Pass structured messages
                images=pil_images,  # Pass list of PIL Images
                return_tensors="pt",
                padding=True        # Enable padding for potentially varying text lengths if batching
            ).to(DEVICE) # Move inputs to the target device
            # Note: dtype conversion should happen automatically based on model or quantization
            logger.info("Model inputs prepared using multi-image approach.")
        except AttributeError as e:
             # Catch the specific error we saw before, just in case
             logger.error(f"AttributeError during processor call (Should be fixed!): {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Internal processor error (AttributeError): {e}")
        except Exception as e:
             logger.error(f"Error during processor call: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Error preparing model inputs: {e}")

        # Free PIL image memory
        del pil_images

        # 3. Setup Streaming
        streamer = TextIteratorStreamer(
            tokenizer,
            timeout=60.0,
            skip_prompt=True, # Set to True - usually don't want prompt in output
            skip_special_tokens=True
        )

        # 4. Run Generation in a Separate Thread
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True if request.temperature > 0 else False, # Enable sampling if temp > 0
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        logger.info("Generation thread started.")

        # 5. Create Async Generator for Streaming Response (same as before)
        async def event_generator() -> AsyncGenerator[str, None]:
            yielded_anything = False
            try:
                for new_text in streamer:
                    if new_text:
                        yield new_text
                        yielded_anything = True
                thread.join() # Wait for thread completion
                if not yielded_anything:
                    logger.warning("Streamer finished without yielding any content.")
                    yield ""
                logger.info("Streamer finished.")
                log_system_info("Stream Complete")
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}", exc_info=True)
                if thread.is_alive(): # Ensure thread is joined on error too
                     thread.join()
                yield f"\n[ERROR: Streaming failed: {str(e)}]"
            finally:
                if thread.is_alive():
                    logger.warning("Generation thread still alive after streamer finished unexpectedly. Joining.")
                    thread.join()
                request_end_time = time.time()
                logger.info(f"Request processed in {request_end_time - request_start_time:.2f} seconds.")

        return StreamingResponse(event_generator(), media_type="text/plain")

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unhandled error in /v1/video/describe: {str(e)}", exc_info=True)
        log_system_info("Request Error")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


# --- Main Execution ---
if __name__ == "__main__":
    # Use environment variables or command-line args for port in production
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
