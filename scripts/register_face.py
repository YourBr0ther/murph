#!/usr/bin/env python3
"""
Register a face with Murph's memory system.

Usage:
    # Capture from webcam and register
    python scripts/register_face.py --name "Chris" --capture

    # Register from an image file
    python scripts/register_face.py --name "Chris" --image photo.jpg

    # List registered people
    python scripts/register_face.py --list
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from server.storage import Database
from server.cognition.memory.long_term_memory import LongTermMemory
from server.cognition.memory.memory_types import PersonMemory
from server.perception.vision.face_detector import FaceDetector
from server.perception.vision.face_encoder import FaceEncoder


async def list_people(ltm: LongTermMemory) -> None:
    """List all registered people."""
    people = await ltm.get_all_people()

    if not people:
        print("No people registered yet.")
        return

    print(f"\nRegistered people ({len(people)}):")
    print("-" * 50)
    for person in people:
        embeddings = await ltm.get_face_embeddings(person.person_id)
        print(f"  {person.name or '(unnamed)'}")
        print(f"    ID: {person.person_id}")
        print(f"    Familiarity: {person.familiarity_score:.0f}/100")
        print(f"    Trust: {person.trust_score:.0f}/100")
        print(f"    Face embeddings: {len(embeddings)}")
        print()


async def capture_face() -> np.ndarray | None:
    """Capture a face from the webcam."""
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None

    print("Position your face in the frame and press SPACE to capture (Q to quit)")

    detector = FaceDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for display info
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = detector.detect(rgb)

        # Draw rectangles around faces
        for face in faces:
            x, y, w, h = int(face.x), int(face.y), int(face.width), int(face.height)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {face.confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show status
        status = f"Faces detected: {len(faces)}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=capture, Q=quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == ord(' ') and len(faces) == 1:
            cap.release()
            cv2.destroyAllWindows()
            return rgb
        elif key == ord(' ') and len(faces) != 1:
            print(f"Need exactly 1 face, found {len(faces)}")

    cap.release()
    cv2.destroyAllWindows()
    return None


async def register_face(
    ltm: LongTermMemory,
    name: str,
    image: np.ndarray,
    familiarity: float = 80.0,
    trust: float = 80.0,
) -> bool:
    """Register a face with the memory system."""
    detector = FaceDetector()
    encoder = FaceEncoder()

    # Detect face
    print("Detecting face...")
    faces = detector.detect(image)

    if not faces:
        print("Error: No face detected in image")
        return False

    if len(faces) > 1:
        print(f"Warning: {len(faces)} faces detected, using the largest one")
        faces = sorted(faces, key=lambda f: f.width * f.height, reverse=True)

    face = faces[0]
    print(f"  Face detected: {face.width}x{face.height} pixels, confidence {face.confidence:.2f}")

    # Encode face
    print("Encoding face...")
    encodings = encoder.encode(image, [face])

    if not encodings:
        print("Error: Failed to encode face")
        return False

    encoding = encodings[0]
    print(f"  Embedding quality: {encoding.quality_score:.2f}")

    # Create person ID from name
    person_id = name.lower().replace(" ", "_") + "_1"

    # Check if person already exists
    existing = await ltm.get_person(person_id)
    if existing:
        print(f"Person '{name}' already exists, adding new face embedding...")
    else:
        # Create person record
        print(f"Creating person record for '{name}'...")
        person = PersonMemory(
            person_id=person_id,
            name=name,
            familiarity_score=familiarity,
            trust_score=trust,
            sentiment=0.5,  # Slightly positive
        )
        await ltm.save_person(person, trust_score=trust)

    # Save face embedding
    print("Saving face embedding...")
    success = await ltm.save_face_embedding(
        person_id=person_id,
        embedding=encoding.embedding,
        quality_score=encoding.quality_score,
    )

    if success:
        print(f"\n✓ Successfully registered '{name}'!")
        print(f"  Person ID: {person_id}")
        print(f"  Familiarity: {familiarity}/100")
        print(f"  Trust: {trust}/100")

        # Show how many embeddings they have now
        embeddings = await ltm.get_face_embeddings(person_id)
        print(f"  Total face embeddings: {len(embeddings)}")
        return True
    else:
        print("Error: Failed to save face embedding")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Register faces with Murph")
    parser.add_argument("--name", type=str, help="Person's name")
    parser.add_argument("--capture", action="store_true", help="Capture from webcam")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--list", action="store_true", help="List registered people")
    parser.add_argument("--familiarity", type=float, default=80.0,
                       help="Familiarity score 0-100 (default: 80)")
    parser.add_argument("--trust", type=float, default=80.0,
                       help="Trust score 0-100 (default: 80)")

    args = parser.parse_args()

    # Initialize database
    db = Database()
    await db.initialize()

    ltm = LongTermMemory(db)
    await ltm.initialize()

    try:
        if args.list:
            await list_people(ltm)
        elif args.name and args.capture:
            image = await capture_face()
            if image is not None:
                await register_face(ltm, args.name, image, args.familiarity, args.trust)
        elif args.name and args.image:
            image_path = Path(args.image)
            if not image_path.exists():
                print(f"Error: Image file not found: {args.image}")
                return

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Error: Could not read image: {args.image}")
                return

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            await register_face(ltm, args.name, image, args.familiarity, args.trust)
        else:
            parser.print_help()
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
