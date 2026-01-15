# client/src/murph_client/main.py
import asyncio
import json
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

from murph_client.config import ClientConfig, GPIOPins
from murph_client.audio.capture import AudioCapture
from murph_client.audio.wakeword import WakeWordDetector
from murph_client.audio.playback import AudioPlayback
from murph_client.motors.driver import MotorDriver
from murph_client.sensors.ultrasonic import UltrasonicSensor


class MurphClient:
    def __init__(self, config: ClientConfig = None):
        self.config = config or ClientConfig()
        self.pins = GPIOPins()
        self.running = False
        self.reconnect_delay = 1  # Start with 1 second

        # Initialize components
        self.audio_capture = AudioCapture(
            sample_rate=self.config.sample_rate,
            chunk_size=self.config.chunk_size,
            device_index=self.config.audio_device_index,
            device_sample_rate=self.config.audio_device_sample_rate,
        )
        self.wakeword = WakeWordDetector()
        self.playback = AudioPlayback()
        self.motors = MotorDriver(max_duty_cycle=self.config.max_duty_cycle)
        self.ultrasonic = UltrasonicSensor(
            trig_pin=self.pins.ULTRASONIC_TRIG,
            echo_pin=self.pins.ULTRASONIC_ECHO,
        )

    async def handle_command(self, command: dict) -> bool:
        """Handle a command from the server. Returns True if audio response expected."""
        cmd_type = command.get("type")

        if cmd_type == "no_speech":
            # No speech detected - just go back to listening
            print("(no speech detected)")
            return False

        elif cmd_type == "error":
            # Server error - play sad beep
            print(f"Server error: {command.get('message', 'unknown')}")
            self.playback.beep(frequency=400, duration=0.3)  # Low sad beep
            return False

        elif cmd_type == "command":
            cmd = command.get("command")
            distance = command.get("distance", 3.0)

            if cmd == "stop":
                self.motors.stop()
            elif cmd in ["forward", "backward", "rotate_left", "rotate_right"]:
                # Check for obstacles before moving forward
                if cmd == "forward" and self.ultrasonic.obstacle_detected(
                    self.config.obstacle_threshold_cm
                ):
                    print("Obstacle detected! Backing up...")
                    self.motors.move_for_distance(
                        "backward", self.config.backup_distance_inches
                    )
                else:
                    self.motors.move_for_distance(cmd, distance)
            return False

        elif cmd_type == "response":
            text = command.get("text", "")
            print(f"Murph: {text}")
            return True  # Expecting audio response

        return False

    async def _play_thinking_sounds(self, stop_event: asyncio.Event):
        """Play R2-D2 chirps until stop_event is set."""
        loop = asyncio.get_event_loop()
        while not stop_event.is_set():
            await loop.run_in_executor(None, self.playback.chirp)
            # Wait a bit between chirps, but check stop_event frequently
            for _ in range(10):  # 1 second total, checking every 100ms
                if stop_event.is_set():
                    break
                await asyncio.sleep(0.1)

    async def _listen_loop(self, ws):
        """Main listening loop - handles wake word and server communication."""
        print("Connected! Listening for wake word...")
        self.reconnect_delay = 1  # Reset on successful connection
        loop = asyncio.get_event_loop()

        while self.running:
            # Listen for wake word - run in executor to not block event loop
            audio_chunk = await loop.run_in_executor(None, self.audio_capture.read)
            if self.wakeword.process(audio_chunk):
                print("Wake word detected!")
                self.wakeword.reset()

                # Say "Yes sir?" to acknowledge
                await loop.run_in_executor(None, self.playback.say_yes_sir)
                await asyncio.sleep(0.1)  # Small gap before recording
                print("Recording... speak now!")

                # Record for a few seconds (run in executor to not block)
                audio_data = await loop.run_in_executor(
                    None, self.audio_capture.read_seconds, 3.0
                )

                # Signal recording complete with beep
                self.playback.beep(frequency=1200, duration=0.1)
                print("Processing...")

                # Send to server
                await ws.send(audio_data.tobytes())

                # Start thinking sounds while waiting for response
                stop_thinking = asyncio.Event()
                thinking_task = asyncio.create_task(
                    self._play_thinking_sounds(stop_thinking)
                )

                try:
                    # Wait for response
                    response = await ws.recv()
                finally:
                    # Stop thinking sounds
                    stop_thinking.set()
                    await thinking_task

                if isinstance(response, str):
                    # JSON response
                    command = json.loads(response)
                    expects_audio = await self.handle_command(command)

                    # Check if there's also audio to play
                    if expects_audio:
                        audio_response = await ws.recv()
                        if isinstance(audio_response, bytes):
                            self.playback.play(audio_response)
                elif isinstance(response, bytes):
                    # Direct audio response
                    self.playback.play(response)

                print("Listening for wake word...")

    async def run(self):
        """Run with automatic reconnection on disconnect."""
        self.running = True

        while self.running:
            try:
                print(f"Connecting to {self.config.server_uri}...")
                async with websockets.connect(
                    self.config.server_uri,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    await self._listen_loop(ws)

            except (ConnectionClosed, ConnectionClosedError) as e:
                print(f"Connection lost: {e}")
                self.playback.beep(frequency=300, duration=0.2)  # Disconnected beep
                await asyncio.sleep(0.1)
                self.playback.beep(frequency=300, duration=0.2)

            except (OSError, TimeoutError) as e:
                print(f"Connection failed: {e}")

            except Exception as e:
                print(f"Unexpected error: {e}")

            if self.running:
                print(f"Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)
                # Exponential backoff, max 30 seconds
                self.reconnect_delay = min(self.reconnect_delay * 2, 30)

    def stop(self):
        self.running = False
        self.motors.stop()

    def cleanup(self):
        self.stop()
        self.audio_capture.close()
        self.playback.cleanup()
        self.motors.cleanup()


async def main():
    client = MurphClient()
    try:
        await client.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
