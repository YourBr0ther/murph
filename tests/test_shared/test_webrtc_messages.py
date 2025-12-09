"""Tests for WebRTC message types."""

import pytest

from shared.messages import (
    MessageType,
    RobotMessage,
    WebRTCAnswer,
    WebRTCIceCandidate,
    WebRTCOffer,
    create_webrtc_answer,
    create_webrtc_ice_candidate,
    create_webrtc_offer,
)


class TestWebRTCMessageTypes:
    """Tests for WebRTC message type enum values."""

    def test_message_type_values(self) -> None:
        """Test that WebRTC message types have correct values."""
        assert MessageType.WEBRTC_OFFER == 40
        assert MessageType.WEBRTC_ANSWER == 41
        assert MessageType.WEBRTC_ICE_CANDIDATE == 42


class TestWebRTCOffer:
    """Tests for WebRTCOffer dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        offer = WebRTCOffer()
        assert offer.sdp == ""
        assert offer.type == "offer"

    def test_with_sdp(self) -> None:
        """Test with SDP content."""
        sdp = "v=0\no=- 12345 1 IN IP4 127.0.0.1\n..."
        offer = WebRTCOffer(sdp=sdp)
        assert offer.sdp == sdp
        assert offer.type == "offer"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        offer = WebRTCOffer(sdp="test_sdp")
        d = offer.to_dict()
        assert d["sdp"] == "test_sdp"
        assert d["type"] == "offer"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {"sdp": "test_sdp", "type": "offer"}
        offer = WebRTCOffer.from_dict(data)
        assert offer.sdp == "test_sdp"
        assert offer.type == "offer"

    def test_from_dict_defaults(self) -> None:
        """Test creation from empty dictionary uses defaults."""
        offer = WebRTCOffer.from_dict({})
        assert offer.sdp == ""
        assert offer.type == "offer"


class TestWebRTCAnswer:
    """Tests for WebRTCAnswer dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        answer = WebRTCAnswer()
        assert answer.sdp == ""
        assert answer.type == "answer"

    def test_with_sdp(self) -> None:
        """Test with SDP content."""
        sdp = "v=0\no=- 67890 1 IN IP4 127.0.0.1\n..."
        answer = WebRTCAnswer(sdp=sdp)
        assert answer.sdp == sdp
        assert answer.type == "answer"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        answer = WebRTCAnswer(sdp="answer_sdp")
        d = answer.to_dict()
        assert d["sdp"] == "answer_sdp"
        assert d["type"] == "answer"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {"sdp": "answer_sdp", "type": "answer"}
        answer = WebRTCAnswer.from_dict(data)
        assert answer.sdp == "answer_sdp"
        assert answer.type == "answer"


class TestWebRTCIceCandidate:
    """Tests for WebRTCIceCandidate dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        candidate = WebRTCIceCandidate()
        assert candidate.candidate == ""
        assert candidate.sdp_mid is None
        assert candidate.sdp_mline_index is None

    def test_with_values(self) -> None:
        """Test with all values."""
        candidate = WebRTCIceCandidate(
            candidate="candidate:1 1 UDP 2013266431 192.168.1.1 5000 typ host",
            sdp_mid="video",
            sdp_mline_index=0,
        )
        assert "UDP" in candidate.candidate
        assert candidate.sdp_mid == "video"
        assert candidate.sdp_mline_index == 0

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        candidate = WebRTCIceCandidate(
            candidate="test_candidate",
            sdp_mid="audio",
            sdp_mline_index=1,
        )
        d = candidate.to_dict()
        assert d["candidate"] == "test_candidate"
        assert d["sdp_mid"] == "audio"
        assert d["sdp_mline_index"] == 1

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "candidate": "test_candidate",
            "sdp_mid": "video",
            "sdp_mline_index": 0,
        }
        candidate = WebRTCIceCandidate.from_dict(data)
        assert candidate.candidate == "test_candidate"
        assert candidate.sdp_mid == "video"
        assert candidate.sdp_mline_index == 0

    def test_from_dict_missing_optional(self) -> None:
        """Test creation with missing optional fields."""
        data = {"candidate": "test_candidate"}
        candidate = WebRTCIceCandidate.from_dict(data)
        assert candidate.candidate == "test_candidate"
        assert candidate.sdp_mid is None
        assert candidate.sdp_mline_index is None


class TestRobotMessageWebRTC:
    """Tests for RobotMessage with WebRTC payloads."""

    def test_webrtc_offer_message(self) -> None:
        """Test creating message with WebRTC offer."""
        msg = RobotMessage(
            message_type=MessageType.WEBRTC_OFFER,
            payload=WebRTCOffer(sdp="offer_sdp"),
        )
        assert msg.message_type == MessageType.WEBRTC_OFFER
        assert isinstance(msg.payload, WebRTCOffer)
        assert msg.payload.sdp == "offer_sdp"

    def test_webrtc_answer_message(self) -> None:
        """Test creating message with WebRTC answer."""
        msg = RobotMessage(
            message_type=MessageType.WEBRTC_ANSWER,
            payload=WebRTCAnswer(sdp="answer_sdp"),
        )
        assert msg.message_type == MessageType.WEBRTC_ANSWER
        assert isinstance(msg.payload, WebRTCAnswer)
        assert msg.payload.sdp == "answer_sdp"

    def test_webrtc_ice_candidate_message(self) -> None:
        """Test creating message with ICE candidate."""
        msg = RobotMessage(
            message_type=MessageType.WEBRTC_ICE_CANDIDATE,
            payload=WebRTCIceCandidate(candidate="test"),
        )
        assert msg.message_type == MessageType.WEBRTC_ICE_CANDIDATE
        assert isinstance(msg.payload, WebRTCIceCandidate)

    def test_serialize_deserialize_offer(self) -> None:
        """Test serialization roundtrip for offer."""
        original = RobotMessage(
            message_type=MessageType.WEBRTC_OFFER,
            payload=WebRTCOffer(sdp="test_offer_sdp"),
        )

        serialized = original.serialize()
        restored = RobotMessage.deserialize(serialized)

        assert restored.message_type == MessageType.WEBRTC_OFFER
        assert isinstance(restored.payload, WebRTCOffer)
        assert restored.payload.sdp == "test_offer_sdp"

    def test_serialize_deserialize_answer(self) -> None:
        """Test serialization roundtrip for answer."""
        original = RobotMessage(
            message_type=MessageType.WEBRTC_ANSWER,
            payload=WebRTCAnswer(sdp="test_answer_sdp"),
        )

        serialized = original.serialize()
        restored = RobotMessage.deserialize(serialized)

        assert restored.message_type == MessageType.WEBRTC_ANSWER
        assert isinstance(restored.payload, WebRTCAnswer)
        assert restored.payload.sdp == "test_answer_sdp"

    def test_serialize_deserialize_ice_candidate(self) -> None:
        """Test serialization roundtrip for ICE candidate."""
        original = RobotMessage(
            message_type=MessageType.WEBRTC_ICE_CANDIDATE,
            payload=WebRTCIceCandidate(
                candidate="test_candidate",
                sdp_mid="video",
                sdp_mline_index=0,
            ),
        )

        serialized = original.serialize()
        restored = RobotMessage.deserialize(serialized)

        assert restored.message_type == MessageType.WEBRTC_ICE_CANDIDATE
        assert isinstance(restored.payload, WebRTCIceCandidate)
        assert restored.payload.candidate == "test_candidate"
        assert restored.payload.sdp_mid == "video"
        assert restored.payload.sdp_mline_index == 0


class TestWebRTCFactoryFunctions:
    """Tests for WebRTC factory helper functions."""

    def test_create_webrtc_offer(self) -> None:
        """Test create_webrtc_offer factory."""
        msg = create_webrtc_offer("test_sdp")
        assert msg.message_type == MessageType.WEBRTC_OFFER
        assert isinstance(msg.payload, WebRTCOffer)
        assert msg.payload.sdp == "test_sdp"

    def test_create_webrtc_answer(self) -> None:
        """Test create_webrtc_answer factory."""
        msg = create_webrtc_answer("test_sdp")
        assert msg.message_type == MessageType.WEBRTC_ANSWER
        assert isinstance(msg.payload, WebRTCAnswer)
        assert msg.payload.sdp == "test_sdp"

    def test_create_webrtc_ice_candidate(self) -> None:
        """Test create_webrtc_ice_candidate factory."""
        msg = create_webrtc_ice_candidate(
            candidate="test_candidate",
            sdp_mid="audio",
            sdp_mline_index=1,
        )
        assert msg.message_type == MessageType.WEBRTC_ICE_CANDIDATE
        assert isinstance(msg.payload, WebRTCIceCandidate)
        assert msg.payload.candidate == "test_candidate"
        assert msg.payload.sdp_mid == "audio"
        assert msg.payload.sdp_mline_index == 1

    def test_create_webrtc_ice_candidate_minimal(self) -> None:
        """Test create_webrtc_ice_candidate with minimal args."""
        msg = create_webrtc_ice_candidate("test_candidate")
        assert msg.message_type == MessageType.WEBRTC_ICE_CANDIDATE
        assert isinstance(msg.payload, WebRTCIceCandidate)
        assert msg.payload.candidate == "test_candidate"
        assert msg.payload.sdp_mid is None
        assert msg.payload.sdp_mline_index is None
