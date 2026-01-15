#!/usr/bin/env python3
"""
Audio Test Script

Plays test tones through the audio output to verify speaker/headphone setup.
Run on the Raspberry Pi with speakers or headphones connected.

Usage:
    python test_audio.py
"""
import sys
import time

try:
    from murph_client.audio.playback import AudioPlayback
except ImportError:
    print("Error: murph_client not installed. Run 'pip install -e client/' first.")
    sys.exit(1)


def main():
    print("=" * 50)
    print("     Murph Audio Test")
    print("=" * 50)
    print()
    print("This script tests audio playback through PyAudio.")
    print("Make sure speakers or headphones are connected.")
    print()

    try:
        playback = AudioPlayback()
    except RuntimeError as e:
        print(f"Error initializing audio: {e}")
        sys.exit(1)

    print("Audio system initialized.")
    print()

    try:
        # Test 1: Simple beep
        print("Test 1: Single beep (800 Hz)...")
        playback.beep(frequency=800, duration=0.3)
        time.sleep(0.5)
        result = input("Did you hear a beep? [y/n]: ").lower()
        if result != 'y':
            print("Check your audio output device and volume settings.")
            return

        # Test 2: Multiple frequencies
        print("\nTest 2: Frequency sweep (low to high)...")
        for freq in [300, 500, 800, 1200, 1600]:
            print(f"  {freq} Hz")
            playback.beep(frequency=freq, duration=0.2)
            time.sleep(0.1)
        time.sleep(0.3)

        # Test 3: Yes sir response
        print("\nTest 3: 'Yes sir?' response...")
        playback.say_yes_sir()
        time.sleep(0.5)

        # Test 4: R2-D2 chirps
        print("\nTest 4: R2-D2 style chirps (3x)...")
        for i in range(3):
            print(f"  Chirp {i+1}")
            playback.chirp()
            time.sleep(0.5)

        # Test 5: Recording indicators
        print("\nTest 5: Recording indicator beeps...")
        print("  Low beep (recording start)...")
        playback.beep(frequency=600, duration=0.15)
        time.sleep(0.3)
        print("  High beep (recording complete)...")
        playback.beep(frequency=1200, duration=0.1)
        time.sleep(0.3)

        print()
        print("=" * 50)
        print("     Audio Test Complete!")
        print("=" * 50)
        print()
        print("If you heard all the sounds, audio is working correctly.")
        print()

    except KeyboardInterrupt:
        print("\n\nTest cancelled.")
    finally:
        playback.cleanup()
        print("Audio cleaned up.")


if __name__ == "__main__":
    main()
