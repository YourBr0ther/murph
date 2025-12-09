// Murph Emulator Frontend

let ws = null;
let reconnectTimer = null;

// DOM Elements
const elements = {
    robot: document.getElementById('robot'),
    position: document.getElementById('position'),
    heading: document.getElementById('heading'),
    expression: document.getElementById('expression'),
    sound: document.getElementById('sound'),
    moving: document.getElementById('moving'),
    state: document.getElementById('state'),
    // Connection status
    uiDot: document.getElementById('uiDot'),
    uiStatus: document.getElementById('uiStatus'),
    serverDot: document.getElementById('serverDot'),
    serverStatus: document.getElementById('serverStatus'),
    // Motor display
    leftWheelBar: document.getElementById('leftWheelBar'),
    rightWheelBar: document.getElementById('rightWheelBar'),
    leftSpeed: document.getElementById('leftSpeed'),
    rightSpeed: document.getElementById('rightSpeed'),
    motorDirection: document.getElementById('motorDirection'),
    // Sensor display
    imuGraph: document.getElementById('imuGraph'),
    accelX: document.getElementById('accelX'),
    accelY: document.getElementById('accelY'),
    accelZ: document.getElementById('accelZ'),
    touchElectrodes: document.getElementById('touchElectrodes'),
};

// IMU Graph class for rolling sensor visualization
class SensorGraph {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.maxPoints = 100;
        this.data = { x: [], y: [], z: [] };
    }

    addPoint(x, y, z) {
        this.data.x.push(x);
        this.data.y.push(y);
        this.data.z.push(z);

        while (this.data.x.length > this.maxPoints) {
            this.data.x.shift();
            this.data.y.shift();
            this.data.z.shift();
        }
        this.render();
    }

    render() {
        const { ctx, canvas } = this;
        const width = canvas.width;
        const height = canvas.height;

        // Clear background
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        // Draw center line (0g)
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // Draw each axis
        this.drawLine(this.data.x, '#ff6b6b'); // Red for X
        this.drawLine(this.data.y, '#4ecdc4'); // Teal for Y
        this.drawLine(this.data.z, '#45b7d1'); // Blue for Z
    }

    drawLine(data, color) {
        if (data.length < 2) return;

        const { ctx, canvas, maxPoints } = this;
        const width = canvas.width;
        const height = canvas.height;
        const scale = height / 6; // -3g to +3g range

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;

        data.forEach((val, i) => {
            const x = (i / maxPoints) * width;
            const y = height / 2 - (val * scale);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.stroke();
    }
}

// Initialize IMU graph
let imuGraph = null;

// WebSocket Connection
function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => {
        console.log('Connected to emulator');
        elements.uiDot.className = 'dot connected';
        elements.uiStatus.textContent = 'Connected';

        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
    };

    ws.onclose = () => {
        console.log('Disconnected from emulator');
        elements.uiDot.className = 'dot disconnected';
        elements.uiStatus.textContent = 'Disconnected';
        elements.serverDot.className = 'dot disconnected';
        elements.serverStatus.textContent = 'Unknown';

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

    // Update connection status
    updateConnectionStatus(state);

    // Update motor display
    updateMotorDisplay(state);

    // Update sensor display
    updateSensorDisplay(state);

    // Update raw state
    elements.state.textContent = JSON.stringify(state, null, 2);
}

// Update sensor data display
function updateSensorDisplay(state) {
    // Update IMU values
    const x = state.imu_accel_x || 0;
    const y = state.imu_accel_y || 0;
    const z = state.imu_accel_z || -1;

    elements.accelX.textContent = `X: ${x.toFixed(2)}`;
    elements.accelY.textContent = `Y: ${y.toFixed(2)}`;
    elements.accelZ.textContent = `Z: ${z.toFixed(2)}`;

    // Update IMU graph
    if (imuGraph) {
        imuGraph.addPoint(x, y, z);
    }

    // Update touch electrodes
    const electrodes = state.touched_electrodes || [];
    if (electrodes.length > 0) {
        elements.touchElectrodes.textContent = electrodes.join(', ');
    } else {
        elements.touchElectrodes.textContent = 'None';
    }
}

// Update server connection status
function updateConnectionStatus(state) {
    if (state.server_connected) {
        elements.serverDot.className = 'dot connected';
        elements.serverStatus.textContent = 'Connected';
    } else {
        elements.serverDot.className = 'dot disconnected';
        elements.serverStatus.textContent = 'Disconnected';
    }
}

// Update motor visualization
function updateMotorDisplay(state) {
    const leftSpeed = state.left_speed || 0;
    const rightSpeed = state.right_speed || 0;

    // Left wheel
    const leftHeight = Math.abs(leftSpeed) * 50; // 50% max height
    elements.leftWheelBar.style.height = `${leftHeight}%`;
    if (leftSpeed < 0) {
        elements.leftWheelBar.className = 'wheel-bar reverse';
    } else {
        elements.leftWheelBar.className = 'wheel-bar';
    }
    elements.leftSpeed.textContent = `${Math.round(leftSpeed * 100)}%`;

    // Right wheel
    const rightHeight = Math.abs(rightSpeed) * 50;
    elements.rightWheelBar.style.height = `${rightHeight}%`;
    if (rightSpeed < 0) {
        elements.rightWheelBar.className = 'wheel-bar reverse';
    } else {
        elements.rightWheelBar.className = 'wheel-bar';
    }
    elements.rightSpeed.textContent = `${Math.round(rightSpeed * 100)}%`;

    // Direction
    elements.motorDirection.textContent = state.move_direction.toUpperCase();
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
    // Create IMU graph
    if (elements.imuGraph) {
        imuGraph = new SensorGraph(elements.imuGraph);
    }
    // Connect to WebSocket
    connect();
});
