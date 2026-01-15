#!/usr/bin/env python3
"""
Motor Calibration Script

Interactive script to measure inches/second at 25% duty cycle.
Run on the Raspberry Pi with the robot on a flat surface.

Usage:
    python calibrate_motors.py
"""
import sys
import time

try:
    from murph_client.motors.driver import MotorDriver
except ImportError:
    print("Error: murph_client not installed. Run 'pip install -e client/' first.")
    sys.exit(1)


def measure_speed(driver: MotorDriver, direction: str, duration: float = 3.0) -> float:
    """
    Run motor in given direction for duration seconds.
    Returns user-measured distance in inches.
    """
    print(f"\nPrepare to measure {direction} movement.")
    print("Place a ruler or tape measure alongside the robot.")
    input("Press Enter when ready...")

    print(f"Running {direction} for {duration} seconds...")
    getattr(driver, direction)()
    time.sleep(duration)
    driver.stop()

    while True:
        try:
            distance = float(input(f"How many inches did the robot move? "))
            return distance / duration
        except ValueError:
            print("Please enter a number.")


def main():
    print("=" * 50)
    print("     Murph Motor Calibration")
    print("=" * 50)
    print()
    print("This script helps calibrate motor speed (inches/second)")
    print("at 25% duty cycle for accurate distance-based movement.")
    print()
    print("Requirements:")
    print("  - Robot on flat surface with room to move")
    print("  - Ruler or tape measure")
    print("  - About 5 minutes")
    print()

    input("Press Enter to begin...")

    try:
        driver = MotorDriver(max_duty_cycle=25)
    except RuntimeError as e:
        print(f"Error initializing motors: {e}")
        print("Make sure you're running this on the Raspberry Pi.")
        sys.exit(1)

    results = {}

    try:
        # Test forward
        ips = measure_speed(driver, "forward")
        results["forward"] = ips
        print(f"  Forward: {ips:.2f} inches/second")

        # Test backward
        ips = measure_speed(driver, "backward")
        results["backward"] = ips
        print(f"  Backward: {ips:.2f} inches/second")

        # Average for linear movement
        avg_linear = (results["forward"] + results["backward"]) / 2

        print()
        print("=" * 50)
        print("     Calibration Results")
        print("=" * 50)
        print()
        print(f"  Forward:  {results['forward']:.2f} in/s")
        print(f"  Backward: {results['backward']:.2f} in/s")
        print(f"  Average:  {avg_linear:.2f} in/s")
        print()
        print("Update your config.py with:")
        print(f"  INCHES_PER_SECOND = {avg_linear:.2f}")
        print()

    except KeyboardInterrupt:
        print("\n\nCalibration cancelled.")
    finally:
        driver.cleanup()
        print("Motors cleaned up.")


if __name__ == "__main__":
    main()
