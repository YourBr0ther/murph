# client/src/murph_client/main.py
import asyncio
import json
import numpy as np
import websockets

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

        # Initialize components
        self.audio_capture = AudioCapture(
            sample_rate=self.config.sample_rate,
            chunk_size=self.config.chunk_size,
        )
        self.wakeword = WakeWordDetector()
        self.playback = AudioPlayback(pin=self.pins.PWM_AUDIO)
        self.motors = MotorDriver(max_duty_cycle=self.config.max_duty_cycle)
        self.ultrasonic = UltrasonicSensor(
            trig_pin=self.pins.ULTRASONIC_TRIG,
            echo_pin=self.pins.ULTRASONIC_ECHO,
        )

    async def handle_command(self, command: dict):
        cmd_type = command.get("type")

        if cmd_type == "command":
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

        elif cmd_type == "response":
            text = command.get("text", "")
            print(f"Murph: {text}")

    async def run(self):
        self.running = True
        print(f"Connecting to {self.config.server_uri}...")

        async with websockets.connect(self.config.server_uri) as ws:
            print("Connected! Listening for wake word...")

            while self.running:
                # Listen for wake word
                audio_chunk = self.audio_capture.read()
                if self.wakeword.process(audio_chunk):
                    print("Wake word detected! Recording...")
                    self.wakeword.reset()

                    # Record for a few seconds
                    audio_data = self.audio_capture.read_seconds(3.0)

                    # Send to server
                    await ws.send(audio_data.tobytes())

                    # Wait for response
                    response = await ws.recv()

                    if isinstance(response, str):
                        # JSON response
                        command = json.loads(response)
                        await self.handle_command(command)

                        # Check if there's also audio to play
                        if command.get("type") == "response":
                            audio_response = await ws.recv()
                            if isinstance(audio_response, bytes):
                                self.playback.play(audio_response)
                    elif isinstance(response, bytes):
                        # Direct audio response
                        self.playback.play(response)

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
