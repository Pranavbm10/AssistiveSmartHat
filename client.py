import cv2
import base64
import requests
import argparse
import time
import sys
import os
import json
import pyttsx3
import threading
import queue
import speech_recognition as sr

# --- Configuration ---
REQUEST_TIMEOUT = 180
DEFAULT_JPEG_QUALITY = 85
DEFAULT_CAMERA_INDEX = 0
DEFAULT_CAPTURE_INTERVAL = 0.5  # Seconds
STABILITY_TOKEN = "STABLE" # Exact token the LLM should return for stable scenes
DEFAULT_REASSURANCE_INTERVAL = 20 # Seconds, after which a reassurance is given if scene remains STABLE
VOICE_COMMAND_KEY = 'v' # Key to press to issue a voice command
MODE_SWITCH_KEY = 'm'   # Key to press to switch modes

# --- REFINED DEFAULT PROMPT for CONCISE, CONVERSATIONAL, and CONTEXTUAL navigation ---
DEFAULT_NAV_PROMPT = (
    "You are my AI assistant helping me, a visually impaired person, navigate. Speak to me directly, concisely, and naturally, like a helpful friend. "
    "Your main goal is to keep me aware of my immediate surroundings and safe for my next few steps. Be extremely quick with your updates.\n\n"
    "First, very briefly, where am I? (e.g., 'Okay, we're on a sidewalk, shops around,' or 'Looks like an indoor hallway.')\n\n"
    "Next, any critical text very close by? (e.g., 'That sign says EXIT.') If none obvious, just say 'No clear signs right here.'\n\n"
    "Most importantly: what is directly in my path for the next few steps? "
    "**If the way directly ahead is blocked by a wall, closed door, large object, person, or anything preventing forward movement within 2-3 steps, state this clearly and immediately. Examples: 'Careful, wall directly ahead.' 'Hold on, closed door in front.' 'Watch out, person blocking path.' 'Big box right ahead.'** "
    "If the path directly ahead *is* clear for a few steps, say so, like 'Path ahead is clear for a bit.' or 'Looking good, path is clear.' "
    "Mention immediate hazards like steps (up or down), curbs, or obstacles to maneuver around. Examples: 'Careful now, step down.' 'Easy, curb on your right.' 'Just a pole on your left, steer a bit right.'\n\n"
    "Focus only on what's new or changed and critical for immediate safety and movement. Keep each update to 1-2 short sentences. Prioritize blockage and hazard warnings above all.\n\n"
    "**CONTEXTUAL AWARENESS:** If you've just described the scene and the very next image shows no significant changes relevant to my immediate path "
    "(no new obstacles, no new important signs, path condition the same, no new hazards), "
    f"then simply respond with the exact phrase '{STABILITY_TOKEN}'. Do not add any other text if the scene is stable. "
    "Otherwise, if there ARE changes, something new to report, or it's a fresh view, provide your normal concise update focusing on what's new or critical."
)

# --- NEW PROMPT for DESCRIPTIVE MODE ---
DEFAULT_DESCRIPTIVE_PROMPT = (
    "You are my AI assistant, and I am visually impaired. You are currently in 'Descriptive Mode'. "
    "Provide a rich and detailed description of the scene in front of me based on the camera view. "
    "Focus on identifying objects, people, the layout of the environment, colors, textures, and the overall atmosphere. "
    "Help me understand what the space looks like and what is happening. Speak naturally and conversationally. "
    "If the scene is largely unchanged from your previous description, you can simply respond with the "
    f"exact phrase '{STABILITY_TOKEN}'. Otherwise, describe the new view in detail."
)

# --- PROMPT for answering a specific voice command ---
VOICE_COMMAND_QUERY_PROMPT_TEMPLATE = (
    "You are an AI assistant viewing a scene for a visually impaired user. "
    "Based *only* on the provided image, answer the following user's question concisely and directly: '{user_question}'. "
    "Do not provide a general scene description unless it's essential to answer the question."
)


# --- TTS Worker Thread ---
class TTSWorker(threading.Thread):
    def __init__(self, command_queue):
        super().__init__(daemon=True)
        self.command_queue = command_queue
        self.engine = None
        self._running = True
        self.name = "TTSWorkerThread"
        self.is_speaking = False

    def run(self):
        try:
            self.engine = pyttsx3.init()
        except Exception as e:
            print(f"TTS Worker: Failed to initialize pyttsx3 engine: {e}. Thread stopping.", file=sys.stderr)
            self._running = False
            return

        if not self.engine:
            print("TTS Worker: Engine not initialized after pyttsx3.init(). Thread stopping.", file=sys.stderr)
            self._running = False
            return

        try:
            self.engine.setProperty('rate', 180)
        except Exception as e:
            print(f"TTS Worker: Error setting engine properties: {e}", file=sys.stderr)

        print("TTS Worker: Initialized and running.")
        while self._running:
            try:
                command, data = self.command_queue.get(timeout=0.1)
                if command == 'speak':
                    if self.engine:
                        self.is_speaking = True
                        try:
                            time.sleep(0.1) # MODIFICATION: Added small delay
                            # self.engine.stop() # Clearing queue in speak() is preferred
                            self.engine.say(data)
                            self.engine.runAndWait()
                        except Exception as e_speak:
                            print(f"TTS Worker: Error during say/runAndWait: {e_speak}", file=sys.stderr)
                        finally:
                            self.is_speaking = False
                elif command == 'shutdown':
                    self._running = False
                    if self.engine:
                        self.engine.stop() # Stop any ongoing speech before exiting loop
                self.command_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS worker error: {e}", file=sys.stderr)
                self.is_speaking = False # Ensure flag is reset on unexpected error in loop
                time.sleep(0.1) # Avoid busy-looping on repeated errors
        print("TTS Worker: Exiting run loop.")

    def speak(self, text):
        if not self._running or not self.engine:
            print(f"TTS Worker: Speak called but not running or engine not init: {text}", file=sys.stderr)
            return
        # Clear only 'speak' commands from the queue to prioritize the new one.
        # Keep other commands like 'shutdown'.
        temp_keep_commands = []
        while not self.command_queue.empty():
            try:
                cmd, dat = self.command_queue.get_nowait()
                if cmd != 'speak': # Keep non-speak commands
                    temp_keep_commands.append((cmd, dat))
                self.command_queue.task_done() # Mark dequeued item as done
            except queue.Empty:
                break
        # Re-add preserved commands
        for cmd_item in temp_keep_commands:
            self.command_queue.put(cmd_item)
        # Add the new speak command
        self.command_queue.put(('speak', text))

    def shutdown(self):
        print("TTS Worker: Shutdown signal received.")
        if self._running:
            self._running = False
            # Clear the queue before putting the shutdown command
            # to prevent processing other commands during shutdown.
            # This helps ensure 'shutdown' is the next processed item if queue was not empty.
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                    self.command_queue.task_done()
                except queue.Empty:
                    break
            self.command_queue.put(('shutdown', None)) # Signal the run loop to terminate

    def is_busy(self):
        if self.is_speaking:
            return True
        if not self.command_queue.empty(): # Check if there are commands to be processed
            # More accurately, check if there's a 'speak' command pending
            # For simplicity, any command makes it "busy" for now.
            return True
        return False


# --- Helper Functions ---
def encode_frame_to_base64(frame, quality=DEFAULT_JPEG_QUALITY):
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        is_success, buffer = cv2.imencode(".jpg", frame, encode_param)
        if not is_success:
            print("Error: cv2.imencode failed.", file=sys.stderr)
            return None
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"Error encoding frame: {e}", file=sys.stderr)
        return None

def send_request_to_server(api_url, base64_frames_list, prompt, max_tokens, temp, top_p):
    payload = {
        "images": base64_frames_list, "prompt": prompt,
        "max_new_tokens": max_tokens, "temperature": temp, "top_p": top_p,
    }
    headers = {"Content-Type": "application/json"}
    full_api_url = f"{api_url}/v1/video/describe"
    full_response_text = ""
    try:
        print(f"[{time.strftime('%H:%M:%S')}] Sending API request (max_tokens: {max_tokens}, temp: {temp}). Prompt: \"{prompt[:100]}...\"")
        response_start_time = time.time()
        response = requests.post(full_api_url, headers=headers, json=payload, stream=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk: full_response_text += chunk
        response_time = time.time() - response_start_time
        stripped_response = full_response_text.strip()
        if stripped_response:
            print(f"[{time.strftime('%H:%M:%S')}] API response received ({response_time:.2f}s): \"{stripped_response}\"")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] API returned empty response ({response_time:.2f}s).")
        return stripped_response
    except requests.exceptions.Timeout:
        print(f"\nError: API request timed out after {REQUEST_TIMEOUT}s.", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"\nError communicating with API server: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\nUnexpected error during request: {e}", file=sys.stderr)
        return None

# --- Speech Recognition Helper ---
def listen_for_speech(recognizer, microphone, tts_worker, listen_timeout=5, phrase_time_limit=10):
    if not isinstance(recognizer, sr.Recognizer) or not isinstance(microphone, sr.Microphone):
        return None

    with microphone as source:
        try:
            print("Adjusting for ambient noise (listen_for_speech)...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print(f"Energy threshold set to: {recognizer.energy_threshold:.2f}")
        except Exception as e:
            print(f"Error adjusting for ambient noise: {e}", file=sys.stderr)

        print("Listening for speech...")
        audio = None
        try:
            audio = recognizer.listen(source, timeout=listen_timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("No speech detected within timeout.")
            return None
        except Exception as e:
            print(f"Error during listen: {e}", file=sys.stderr)
            return None

    if audio:
        try:
            with open("debug_recorded_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())
            print("DEBUG: Recorded audio saved to debug_recorded_audio.wav")
        except Exception as e_save:
            print(f"DEBUG: Error saving audio: {e_save}", file=sys.stderr)

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"Error during recognition: {e}", file=sys.stderr)
        return None


# --- Mode Selection Function (Initial) ---
def select_operation_mode(tts_worker, recognizer, microphone):
    global DEFAULT_NAV_PROMPT, DEFAULT_DESCRIPTIVE_PROMPT
    selected_prompt = DEFAULT_NAV_PROMPT
    mode_name = "Navigation Guidance"

    tts_worker.speak("Welcome! Please choose an operation mode. Say 'Navigation' for guidance, or 'Descriptive' for detailed scene descriptions.")
    print("Mode selection: Waiting for welcome message to finish speaking...")
    while tts_worker.is_busy():
        time.sleep(0.1)
    print("Mode selection: Welcome message finished. Now listening for user choice.")

    attempt_count = 0
    max_attempts = 2

    while attempt_count < max_attempts:
        print(f"Mode selection attempt {attempt_count + 1}...")
        if recognizer and microphone:
            choice = listen_for_speech(recognizer, microphone, tts_worker, listen_timeout=7, phrase_time_limit=10)
        else:
            choice = None
            print("Mode selection: Speech recognizer not available for choice.")

        if choice:
            if "nav" in choice:
                selected_prompt = DEFAULT_NAV_PROMPT
                mode_name = "Navigation Guidance"
                tts_worker.speak(f"Navigation Guidance mode selected.")
                print(f"Mode selected: {mode_name}. Waiting for confirmation TTS...")
                while tts_worker.is_busy(): time.sleep(0.1)
                print("Confirmation TTS finished.")
                return mode_name, selected_prompt
            elif "desc" in choice:
                selected_prompt = DEFAULT_DESCRIPTIVE_PROMPT
                mode_name = "Descriptive"
                tts_worker.speak(f"Descriptive mode selected.")
                print(f"Mode selected: {mode_name}. Waiting for confirmation TTS...")
                while tts_worker.is_busy(): time.sleep(0.1)
                print("Confirmation TTS finished.")
                return mode_name, selected_prompt
            else:
                tts_worker.speak("I didn't catch that. Please say 'Navigation' or 'Descriptive'.")
                print("Mode selection: Unrecognized choice. Waiting for re-prompt TTS...")
                while tts_worker.is_busy(): time.sleep(0.1)
                print("Re-prompt TTS finished.")
        else:
            if attempt_count < max_attempts -1:
                 tts_worker.speak("I didn't hear a response. Please try saying 'Navigation' or 'Descriptive'.")
                 print("Mode selection: No response. Waiting for re-prompt TTS...")
                 while tts_worker.is_busy(): time.sleep(0.1)
                 print("Re-prompt TTS finished.")

        attempt_count += 1
        if attempt_count < max_attempts and (recognizer and microphone):
            print("Mode selection: Pausing briefly before next listen attempt.")
            time.sleep(0.5)
        elif not (recognizer and microphone) and attempt_count < max_attempts :
             print("Mode selection: Speech recognizer not available, cannot retry with voice.")
             break

    tts_worker.speak(f"Defaulting to Navigation Guidance mode.")
    print(f"Defaulting to mode: {mode_name}. Waiting for default message TTS...")
    while tts_worker.is_busy(): time.sleep(0.1)
    print("Default message TTS finished.")
    return mode_name, selected_prompt

# --- Function to handle mode switching during runtime ---
def switch_mode_interactive(tts_worker, recognizer, microphone, current_mode_name, current_base_prompt, args_ref):
    global DEFAULT_NAV_PROMPT, DEFAULT_DESCRIPTIVE_PROMPT

    original_max_tokens = args_ref.max_tokens
    original_temp = args_ref.temp

    tts_worker.speak("Mode change. Say 'Navigation' or 'Descriptive' for the new mode, or 'Cancel' to keep the current mode.")
    print("Mode change: Waiting for prompt to finish...")
    while tts_worker.is_busy():
        time.sleep(0.1)
    print("Mode change: Prompt finished. Listening for new mode selection...")

    new_mode_name = current_mode_name
    new_base_prompt = current_base_prompt

    choice = None
    if recognizer and microphone:
        choice = listen_for_speech(recognizer, microphone, tts_worker, listen_timeout=7, phrase_time_limit=10)
    else:
        tts_worker.speak("Speech recognition not available to change mode.")
        while tts_worker.is_busy(): time.sleep(0.1)
        return current_mode_name, current_base_prompt, False

    changed_mode = False
    if choice:
        if "cancel" in choice:
            tts_worker.speak("Mode change cancelled.")
            changed_mode = False
        elif "nav" in choice:
            if current_mode_name == "Navigation Guidance":
                tts_worker.speak("Already in Navigation Guidance mode.")
            else:
                new_mode_name = "Navigation Guidance"
                new_base_prompt = DEFAULT_NAV_PROMPT
                args_ref.max_tokens = 60
                args_ref.temp = 0.3
                tts_worker.speak("Switched to Navigation Guidance mode.")
                changed_mode = True
        elif "desc" in choice:
            if current_mode_name == "Descriptive":
                tts_worker.speak("Already in Descriptive mode.")
            else:
                new_mode_name = "Descriptive"
                new_base_prompt = DEFAULT_DESCRIPTIVE_PROMPT
                args_ref.max_tokens = 200
                args_ref.temp = 0.7
                tts_worker.speak("Switched to Descriptive mode.")
                changed_mode = True
        else:
            tts_worker.speak("Did not understand the mode. Keeping current mode.")
            changed_mode = False
    else:
        tts_worker.speak("No mode specified. Keeping current mode.")
        changed_mode = False

    print(f"Mode change: Waiting for confirmation/cancellation TTS...")
    while tts_worker.is_busy():
        time.sleep(0.1)
    print("Mode change: TTS finished.")

    if not changed_mode:
        args_ref.max_tokens = original_max_tokens
        args_ref.temp = original_temp
        print(f"Mode change cancelled or failed. Reverted params: Tokens: {args_ref.max_tokens}, Temp: {args_ref.temp}")


    if changed_mode:
        print(f"Mode switched to: {new_mode_name}. Max Tokens: {args_ref.max_tokens}, Temp: {args_ref.temp}")
    else:
        print(f"Mode remains: {current_mode_name}.")

    return new_mode_name, new_base_prompt, changed_mode

# --- Main Camera Processing Function ---
def process_live_camera_feed(args, tts_worker, current_mode_name, current_base_prompt, recognizer, microphone):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        error_msg = f"Error: Could not open camera {args.camera_index}"
        print(error_msg, file=sys.stderr)
        if tts_worker and tts_worker.is_alive() and tts_worker.engine: tts_worker.speak(error_msg)
        return

    print(f"Camera {args.camera_index} opened. Mode: {current_mode_name}. 'q': Quit, '{VOICE_COMMAND_KEY}': Voice Cmd, '{MODE_SWITCH_KEY}': Switch Mode")
    print(f"Target API request interval: {args.interval}s. Stability token: '{STABILITY_TOKEN}'. Reassurance interval: {args.reassurance_interval}s.")
    if args.resize_width and args.resize_height: print(f"Frame resizing: {args.resize_width}x{args.resize_height}")

    if tts_worker and tts_worker.is_alive() and tts_worker.engine:
        tts_worker.speak(f"Camera {args.camera_index} activated in {current_mode_name} mode.")
        print("Waiting for camera activation announcement...")
        while tts_worker.is_busy(): time.sleep(0.1)
        print("Camera announcement finished.")
    else: print("Warning: TTS worker not fully active for camera activation message.", file=sys.stderr)

    cv2.namedWindow('Live Camera Feed - Q:Quit, V:Voice, M:Mode', cv2.WINDOW_NORMAL)
    last_spoken_description = ""
    last_api_request_initiation_time = 0
    last_time_meaningful_speech = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame. Attempting to re-open camera...", file=sys.stderr)
                cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(args.camera_index)
                if not cap.isOpened():
                    error_msg = f"Fatal: Could not re-open camera {args.camera_index}."
                    print(error_msg, file=sys.stderr)
                    if tts_worker and tts_worker.is_alive() and tts_worker.engine: tts_worker.speak(error_msg)
                    break
                print("Camera re-opened successfully.")
                continue

            display_frame = frame.copy()
            cv2.putText(display_frame, f"Mode: {current_mode_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Live Camera Feed - Q:Quit, V:Voice, M:Mode', display_frame)

            current_time = time.time()
            key_pressed = cv2.waitKey(30) & 0xFF

            if key_pressed != 255 and key_pressed != -1 :
                try:
                    char_pressed = chr(key_pressed)
                except ValueError:
                    char_pressed = "N/A (non-ASCII)"
                print(f"[{time.strftime('%H:%M:%S')}] cv2.waitKey raw: {key_pressed}, char: '{char_pressed}'")


            if key_pressed == ord('q'):
                print("Quit signal ('q') received.")
                if tts_worker and tts_worker.is_alive() and tts_worker.engine: tts_worker.speak("Exiting program.")
                break

            elif key_pressed == ord(VOICE_COMMAND_KEY):
                if not (recognizer and microphone):
                    if tts_worker and tts_worker.is_alive() and tts_worker.engine:
                        tts_worker.speak("Voice command system is not available.")
                    print("Voice command key pressed, but recognizer/microphone not available.")
                    last_api_request_initiation_time = time.time()
                    continue

                print(f"'{VOICE_COMMAND_KEY}' pressed. Listening for voice command...")
                if tts_worker and tts_worker.is_alive() and tts_worker.engine:
                    if tts_worker.is_speaking: tts_worker.engine.stop() # Interrupt current general speech
                    tts_worker.speak("How can I help?")
                    print("Waiting for 'How can I help?' TTS...")
                    while tts_worker.is_busy(): time.sleep(0.1)
                    print("'How can I help?' TTS finished.")

                user_command_text = listen_for_speech(recognizer, microphone, tts_worker, listen_timeout=7, phrase_time_limit=12)

                if user_command_text:
                    if tts_worker and tts_worker.is_alive() and tts_worker.engine:
                        if tts_worker.is_speaking: tts_worker.engine.stop() # Interrupt "How can I help" if it was cut short by quick command
                        tts_worker.speak(f"Okay, processing your request about: {user_command_text}")
                        print(f"Waiting for 'Okay, processing...' TTS...")
                        while tts_worker.is_busy(): time.sleep(0.1)
                        print("'Okay, processing...' TTS finished.")

                    frame_for_command = frame
                    if args.resize_width and args.resize_height:
                         frame_for_command = cv2.resize(frame, (args.resize_width, args.resize_height), interpolation=cv2.INTER_AREA)
                    b64_frame_command = encode_frame_to_base64(frame_for_command, quality=args.quality)

                    if b64_frame_command:
                        command_prompt = VOICE_COMMAND_QUERY_PROMPT_TEMPLATE.format(user_question=user_command_text)
                        command_response = send_request_to_server(
                            args.api_url, [b64_frame_command], command_prompt,
                            args.max_tokens + 20,
                            args.temp, args.top_p
                        )
                        if command_response:
                            print(f"[{time.strftime('%H:%M:%S')}] Voice command response: \"{command_response}\"")
                            if tts_worker and tts_worker.is_alive() and tts_worker.engine:
                                if tts_worker.is_speaking: tts_worker.engine.stop() # Interrupt "Okay, processing..."
                                tts_worker.speak(command_response)
                        else:
                            no_answer_msg = "Sorry, I couldn't get an answer for that."
                            print(f"[{time.strftime('%H:%M:%S')}] {no_answer_msg}")
                            if tts_worker and tts_worker.is_alive() and tts_worker.engine: tts_worker.speak(no_answer_msg)
                    else:
                        encode_fail_msg = "Could not process the image for your command."
                        print(encode_fail_msg, file=sys.stderr)
                        if tts_worker and tts_worker.is_alive() and tts_worker.engine: tts_worker.speak(encode_fail_msg)
                else:
                    no_command_msg = "I didn't catch your command. Please try again."
                    if tts_worker and tts_worker.is_alive() and tts_worker.engine:
                         tts_worker.speak(no_command_msg)

                last_spoken_description = ""
                last_api_request_initiation_time = time.time()
                last_time_meaningful_speech = time.time()
                continue

            elif key_pressed == ord(MODE_SWITCH_KEY):
                print(f"'{MODE_SWITCH_KEY}' pressed. Initiating mode switch...")
                if tts_worker and tts_worker.is_alive() and tts_worker.engine and tts_worker.is_speaking:
                    tts_worker.engine.stop() # Stop any current speech

                new_mode, new_prompt, mode_was_switched = switch_mode_interactive(
                    tts_worker, recognizer, microphone, current_mode_name, current_base_prompt, args
                )
                if mode_was_switched:
                    current_mode_name = new_mode
                    current_base_prompt = new_prompt
                    print(f"Mode successfully updated to {current_mode_name}.")
                    # Update display immediately, will also be updated at start of loop
                    # display_frame = frame.copy() # Re-copy to avoid drawing over old text if frame didn't change
                    # cv2.putText(display_frame, f"Mode: {current_mode_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # cv2.imshow('Live Camera Feed - Q:Quit, V:Voice, M:Mode', display_frame)
                else:
                    print("Mode change aborted or failed. Continuing in current mode.")

                last_spoken_description = ""
                last_api_request_initiation_time = time.time()
                last_time_meaningful_speech = time.time()
                continue

            if current_time - last_api_request_initiation_time >= args.interval:
                if tts_worker.is_busy():
                    time.sleep(0.05)
                else:
                    last_api_request_initiation_time = current_time

                    frame_to_process = frame
                    if args.resize_width and args.resize_height:
                        frame_to_process = cv2.resize(frame, (args.resize_width, args.resize_height), interpolation=cv2.INTER_AREA)

                    b64_frame = encode_frame_to_base64(frame_to_process, quality=args.quality)
                    if b64_frame:
                        description = send_request_to_server(
                            args.api_url, [b64_frame], current_base_prompt,
                            args.max_tokens, args.temp, args.top_p
                        )

                        if description == STABILITY_TOKEN:
                            if args.reassurance_interval > 0 and \
                               current_time - last_time_meaningful_speech > args.reassurance_interval:
                                reassurance_msg = "All clear, still monitoring."
                                if not tts_worker.is_busy():
                                    print(f"[{time.strftime('%H:%M:%S')}] Speaking reassurance: \"{reassurance_msg}\"")
                                    if tts_worker and tts_worker.is_alive() and tts_worker.engine:
                                        tts_worker.speak(reassurance_msg)
                                    last_time_meaningful_speech = current_time
                        elif description:
                            normalized_new_desc = ' '.join(description.strip().lower().split())
                            normalized_last_desc = ' '.join(last_spoken_description.strip().lower().split())
                            if normalized_new_desc and normalized_new_desc != normalized_last_desc:
                                print(f"[{time.strftime('%H:%M:%S')}] New description from regular update, speaking.")
                                if tts_worker and tts_worker.is_alive() and tts_worker.engine:
                                    if tts_worker.is_speaking: tts_worker.engine.stop() # Stop previous before new
                                    tts_worker.speak(description)
                                last_spoken_description = description
                                last_time_meaningful_speech = current_time
                    # else b64_frame was None (encoding error) - already printed

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
        if tts_worker and tts_worker.is_alive() and tts_worker.engine: tts_worker.speak("Exiting program.")
    finally:
        print("Releasing camera, closing windows...")
        if cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Contextual live camera navigation and description assistant with voice commands.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--api_url", required=True, help="Base URL of the FastAPI/Qwen server")
    parser.add_argument("--prompt", default="AUTO", help=f"Base prompt for the model. Default is 'AUTO' for mode selection. Can be set to one of the predefined prompt strings to override.")
    parser.add_argument("--camera_index", type=int, default=DEFAULT_CAMERA_INDEX, help=f"Camera index (default: {DEFAULT_CAMERA_INDEX})")
    parser.add_argument("--interval", type=float, default=DEFAULT_CAPTURE_INTERVAL, help=f"API request interval for regular updates (s) (default: {DEFAULT_CAPTURE_INTERVAL})")
    parser.add_argument("--max_tokens", type=int, default=150, help="Max new tokens for model (default: 150 for nav, 200 for desc)") # Default will be overridden by mode
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature (default: 0.3 for nav, 0.7 for desc)") # Default will be overridden by mode
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p (default: 0.9)")
    parser.add_argument("--quality", type=int, default=DEFAULT_JPEG_QUALITY, choices=range(0, 101), metavar="[0-100]", help=f"JPEG quality (default: {DEFAULT_JPEG_QUALITY})")
    parser.add_argument("--resize_width", type=int, default=None, help="Frame resize width (default: None)")
    parser.add_argument("--resize_height", type=int, default=None, help="Frame resize height (default: None)")
    parser.add_argument("--reassurance_interval", type=int, default=DEFAULT_REASSURANCE_INTERVAL,
                        help=f"Interval (s) for reassurance speech during stable scenes (0 to disable) (default: {DEFAULT_REASSURANCE_INTERVAL})")
    args = parser.parse_args()

    if not args.api_url:
        print("Error: --api_url is required.", file=sys.stderr); parser.print_help(); sys.exit(1)
    if (args.resize_width and not args.resize_height) or (not args.resize_width and args.resize_height):
        print("Error: Both --resize_width and --resize_height must be specified.", file=sys.stderr); sys.exit(1)

    tts_command_queue = queue.Queue()
    tts_worker_instance = TTSWorker(tts_command_queue)
    tts_worker_instance.start()

    print("Waiting for TTS engine to initialize...")
    initial_tts_wait_start = time.time()
    while not (tts_worker_instance.is_alive() and hasattr(tts_worker_instance, 'engine') and tts_worker_instance.engine):
        time.sleep(0.2)
        if time.time() - initial_tts_wait_start > 10: # Increased timeout
            print("TTS engine timed out during initialization.", file=sys.stderr)
            break

    if not (tts_worker_instance.is_alive() and hasattr(tts_worker_instance, 'engine') and tts_worker_instance.engine):
        print("CRITICAL: TTS worker/engine failed to initialize. Audio output will not work.", file=sys.stderr)
    else:
        print("TTS worker active and engine initialized.")
        tts_worker_instance.speak("System initializing.")
        while tts_worker_instance.is_busy(): time.sleep(0.1)

    recognizer = None
    microphone = None
    try:
        print("Initializing speech recognition...")
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        with microphone as source:
            print("Adjusting microphone for ambient noise (initial setup)...")
            recognizer.adjust_for_ambient_noise(source, duration=1) # Standardized
        print("Speech recognition initialized and microphone adjusted.")
        if tts_worker_instance.is_alive() and tts_worker_instance.engine:
             tts_worker_instance.speak("Speech recognition active.")
             while tts_worker_instance.is_busy(): time.sleep(0.1)
    except AttributeError as e: # Often PyAudio not found or microphone issues
        print(f"Error initializing speech recognition (AttributeError, often PyAudio/Microphone): {e}", file=sys.stderr)
        if tts_worker_instance.is_alive() and tts_worker_instance.engine:
            tts_worker_instance.speak("Speech recognition system failed. Voice commands disabled.")
            while tts_worker_instance.is_busy(): time.sleep(0.1)
        recognizer, microphone = None, None
    except Exception as e: # Catch other potential sr exceptions
        print(f"Could not initialize speech recognition: {e}", file=sys.stderr)
        if tts_worker_instance.is_alive() and tts_worker_instance.engine:
            tts_worker_instance.speak("Speech recognition failed. Voice commands disabled.")
            while tts_worker_instance.is_busy(): time.sleep(0.1)
        recognizer, microphone = None, None

    current_mode_name_main = "Navigation Guidance" # Default
    selected_prompt_main = DEFAULT_NAV_PROMPT     # Default

    # Store whether user explicitly set max_tokens/temp via CLI
    user_set_max_tokens = (args.max_tokens != parser.get_default('max_tokens'))
    user_set_temp = (args.temp != parser.get_default('temp'))


    if args.prompt == "AUTO":
        if recognizer and microphone:
            current_mode_name_main, selected_prompt_main = select_operation_mode(tts_worker_instance, recognizer, microphone)
        else:
            print("Speech recognition not available for automatic mode selection. Defaulting to Navigation Guidance.")
            if tts_worker_instance.is_alive() and tts_worker_instance.engine:
                tts_worker_instance.speak("Defaulting to Navigation Guidance mode as voice selection is unavailable.")
                while tts_worker_instance.is_busy(): time.sleep(0.1)
            # current_mode_name_main and selected_prompt_main are already Nav by default
    elif args.prompt == DEFAULT_DESCRIPTIVE_PROMPT: # Check against actual prompt content
        current_mode_name_main = "Descriptive"
        selected_prompt_main = DEFAULT_DESCRIPTIVE_PROMPT
        print(f"Using specified Descriptive mode via --prompt.")
        if tts_worker_instance.is_alive() and tts_worker_instance.engine:
            tts_worker_instance.speak("Descriptive mode active.")
            while tts_worker_instance.is_busy(): time.sleep(0.1)
    elif args.prompt == DEFAULT_NAV_PROMPT: # Check against actual prompt content
        current_mode_name_main = "Navigation Guidance"
        selected_prompt_main = DEFAULT_NAV_PROMPT
        print(f"Using specified Navigation Guidance mode via --prompt.")
        if tts_worker_instance.is_alive() and tts_worker_instance.engine:
            tts_worker_instance.speak("Navigation guidance mode active.")
            while tts_worker_instance.is_busy(): time.sleep(0.1)
    else: # Custom prompt string
        current_mode_name_main = "Custom Prompt"
        selected_prompt_main = args.prompt
        print(f"Using custom prompt string. Mode treated as general purpose.")
        if tts_worker_instance.is_alive() and tts_worker_instance.engine:
            tts_worker_instance.speak("Custom prompt mode active.")
            while tts_worker_instance.is_busy(): time.sleep(0.1)

    # Set tokens and temp based on the determined mode, ONLY if user didn't specify them via CLI
    if current_mode_name_main == "Navigation Guidance":
        if not user_set_max_tokens: args.max_tokens = 60
        if not user_set_temp: args.temp = 0.3
    elif current_mode_name_main == "Descriptive":
        if not user_set_max_tokens: args.max_tokens = 200
        if not user_set_temp: args.temp = 0.7
    # If Custom Prompt, it will use the CLI-provided values or the initial defaults if not set by CLI.

    print(f"Final initial mode: {current_mode_name_main}, Max Tokens: {args.max_tokens}, Temp: {args.temp}")


    if tts_worker_instance.is_alive() and tts_worker_instance.engine:
        tts_worker_instance.speak("System ready.")
        while tts_worker_instance.is_busy(): time.sleep(0.1)
    else:
        print("System ready (TTS not fully operational).")

    try:
        process_live_camera_feed(args, tts_worker_instance, current_mode_name_main, selected_prompt_main, recognizer, microphone)
    except Exception as e:
        print(f"Unexpected error in main process: {e}", file=sys.stderr)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"Error details: {exc_type}, {fname}, line {exc_tb.tb_lineno}", file=sys.stderr)
        if tts_worker_instance and tts_worker_instance.is_alive() and tts_worker_instance.engine:
             tts_worker_instance.speak("A critical error occurred. Exiting program.")
             # Wait for error message to be spoken if possible
             wait_start = time.time()
             while tts_worker_instance.is_busy() and (time.time() - wait_start < 3): time.sleep(0.1)
    finally:
        print("\nShutting down TTS worker...")
        if tts_worker_instance:
            if tts_worker_instance.is_alive() and tts_worker_instance.is_busy():
                print("Waiting for final TTS messages to clear...")
                wait_start = time.time()
                while tts_worker_instance.is_busy() and (time.time() - wait_start < 3): # Max 3s wait
                    time.sleep(0.1)
            tts_worker_instance.shutdown() # Sends shutdown command
            tts_worker_instance.join(timeout=5) # Wait for thread to exit
            if tts_worker_instance.is_alive():
                print("TTS worker did not shut down cleanly.", file=sys.stderr)
            else:
                print("TTS worker shut down.")
        else:
            print("TTS worker instance was not available for shutdown.")
    print("Script finished.")