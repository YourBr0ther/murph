// Murph Emulator Frontend

let ws = null;
let reconnectTimer = null;

// DOM Elements
const elements = {
    connection: document.getElementById('connection'),
    robot: document.getElementById('robot'),
    position: document.getElementById('position'),
    heading: document.getElementById('heading'),
    expression: document.getElementById('expression'),
    sound: document.getElementById('sound'),
    moving: document.getElementById('moving'),
    state: document.getElementById('state'),
};

// WebSocket Connection
function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => {
        console.log('Connected to emulator');
        elements.connection.textContent = 'Connected';
        elements.connection.className = 'status connected';

        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
    };

    ws.onclose = () => {
        console.log('Disconnected from emulator');
        elements.connection.textContent = 'Disconnected';
        elements.connection.className = 'status disconnected';

        // Reconnect after delay
        reconnectTimer = setTimeout(connect, 2000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'state') {
                updateUI(msg.data);
            }
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    };
}

// Update UI with robot state
function updateUI(state) {
    // Update stats
    elements.position.textContent = `(${state.x.toFixed(1)}, ${state.y.toFixed(1)})`;
    elements.heading.textContent = `${state.heading.toFixed(0)}°`;
    elements.expression.textContent = state.current_expression;
    elements.sound.textContent = state.playing_sound || '-';
    elements.moving.textContent = state.is_moving ? 'Yes' : 'No';

    // Update robot visual
    updateRobotVisual(state);

    // Update raw state
    elements.state.textContent = JSON.stringify(state, null, 2);
}

function updateRobotVisual(state) {
    const robot = elements.robot;

    // Position and rotation
    // Scale position to container (assuming container is 400x300)
    const containerWidth = robot.parentElement.clientWidth;
    const containerHeight = robot.parentElement.clientHeight;

    // Center position with wrapping
    const scale = 2; // pixels per unit
    let x = (containerWidth / 2) + (state.x * scale);
    let y = (containerHeight / 2) - (state.y * scale); // Invert Y for screen coords

    // Wrap around
    x = ((x % containerWidth) + containerWidth) % containerWidth;
    y = ((y % containerHeight) + containerHeight) % containerHeight;

    robot.style.left = `${x}px`;
    robot.style.top = `${y}px`;
    robot.style.transform = `translate(-50%, -50%) rotate(${state.heading}deg)`;

    // Expression classes
    robot.className = 'robot';
    if (state.current_expression !== 'neutral') {
        robot.classList.add(state.current_expression);
    }

    // Moving animation
    if (state.is_moving) {
        robot.classList.add('moving');
    }
}

// Simulation controls
function sendCommand(type, data = {}) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type, ...data }));
    }
}

function simulateTouch() {
    sendCommand('touch', { electrodes: [3, 4, 5, 6] });
    flashButton(event.target);
}

function simulateRelease() {
    sendCommand('release');
    flashButton(event.target);
}

function simulatePickup() {
    sendCommand('pickup');
    flashButton(event.target);
}

function simulateBump() {
    sendCommand('bump');
    flashButton(event.target);

    // Visual feedback on robot
    const robot = elements.robot;
    robot.classList.add('shaking');
    setTimeout(() => robot.classList.remove('shaking'), 300);
}

function simulateShake() {
    sendCommand('shake');
    flashButton(event.target);

    // Visual feedback on robot
    const robot = elements.robot;
    robot.classList.add('shaking');
    setTimeout(() => robot.classList.remove('shaking'), 1000);
}

function flashButton(btn) {
    if (!btn) return;
    btn.style.transform = 'scale(0.95)';
    setTimeout(() => {
        btn.style.transform = '';
    }, 100);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    connect();
});
