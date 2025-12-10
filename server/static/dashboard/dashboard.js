/**
 * Murph Dashboard - Real-time monitoring client
 */

// WebSocket connection
let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 10;
const RECONNECT_DELAY = 2000;

// DOM element cache
const elements = {
    // Connection indicators
    serverDot: document.getElementById('server-dot'),
    piDot: document.getElementById('pi-dot'),
    wsDot: document.getElementById('ws-dot'),

    // Status bar
    currentBehavior: document.getElementById('current-behavior'),
    behaviorState: document.getElementById('behavior-state'),
    behaviorTime: document.getElementById('behavior-time'),
    happinessFill: document.getElementById('happiness-fill'),
    happinessValue: document.getElementById('happiness-value'),
    moodBadge: document.getElementById('mood-badge'),
    urgentNeed: document.getElementById('urgent-need'),

    // Panels
    needsList: document.getElementById('needs-list'),
    behaviorSuggestions: document.getElementById('behavior-suggestions'),
    behaviorSelect: document.getElementById('behavior-select'),
    triggerBtn: document.getElementById('trigger-btn'),
    triggersList: document.getElementById('triggers-list'),
    personCard: document.getElementById('person-card'),
    personName: document.getElementById('person-name'),
    personDetails: document.getElementById('person-details'),
    envConditions: document.getElementById('env-conditions'),
    physicalState: document.getElementById('physical-state'),
    historyList: document.getElementById('history-list'),
    spatialInfo: document.getElementById('spatial-info'),
    timingInfo: document.getElementById('timing-info'),
};

// Need display order and thresholds
const NEEDS_ORDER = ['energy', 'curiosity', 'play', 'social', 'affection', 'comfort', 'safety'];
const CRITICAL_THRESHOLD = 30;
const WARNING_THRESHOLD = 60;

/**
 * Connect to WebSocket server
 */
function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/monitor`;

    console.log('Connecting to', wsUrl);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        reconnectAttempts = 0;
        elements.wsDot.classList.add('connected');
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        elements.wsDot.classList.remove('connected');

        // Auto-reconnect
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            console.log(`Reconnecting (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`);
            setTimeout(connect, RECONNECT_DELAY);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            handleMessage(message);
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    };
}

/**
 * Handle incoming WebSocket messages
 */
function handleMessage(message) {
    if (message.type === 'state') {
        updateDashboard(message.data);
    } else if (message.type === 'behavior_change') {
        console.log('Behavior changed:', message.data);
    } else if (message.type === 'ping') {
        // Respond to ping with pong
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'pong' }));
        }
    }
}

/**
 * Update all dashboard elements with new state
 */
function updateDashboard(state) {
    updateConnectionStatus(state);
    updateStatusBar(state);
    updateNeedsPanel(state.needs);
    updateBehaviorsPanel(state.behaviors, state.available_behaviors);
    updateTriggersPanel(state.world_context);
    updatePerceptionPanel(state.world_context);
    updateHistoryPanel(state.behaviors?.history);
    updateContextPanel(state);
}

/**
 * Update connection status indicators
 */
function updateConnectionStatus(state) {
    if (state.running) {
        elements.serverDot.classList.add('connected');
    } else {
        elements.serverDot.classList.remove('connected');
    }

    if (state.pi_connected) {
        elements.piDot.classList.add('connected');
    } else {
        elements.piDot.classList.remove('connected');
    }
}

/**
 * Update status bar (behavior, happiness, mood)
 */
function updateStatusBar(state) {
    // Current behavior
    elements.currentBehavior.textContent = state.current_behavior || 'idle';
    elements.behaviorState.textContent = state.executor_state || 'IDLE';
    elements.behaviorTime.textContent = `${(state.elapsed_time || 0).toFixed(1)}s`;

    // Happiness gauge
    const happiness = state.needs?.happiness || 0;
    elements.happinessFill.style.width = `${happiness}%`;
    elements.happinessValue.textContent = Math.round(happiness);

    // Mood badge
    const mood = state.needs?.mood || 'neutral';
    elements.moodBadge.textContent = mood;
    elements.moodBadge.className = `mood-badge ${mood}`;

    // Most urgent need
    elements.urgentNeed.textContent = state.needs?.most_urgent || '-';
}

/**
 * Update needs panel with gauges and controls
 */
function updateNeedsPanel(needs) {
    if (!needs?.needs) return;

    const needsData = needs.needs;
    let html = '';

    for (const name of NEEDS_ORDER) {
        const value = needsData[name];
        if (value === undefined) continue;

        const roundedValue = Math.round(value);
        const barClass = value < CRITICAL_THRESHOLD ? 'critical' :
                        value < WARNING_THRESHOLD ? 'warning' : 'good';

        html += `
            <div class="need-item">
                <div class="need-header">
                    <span class="need-name">${name}</span>
                    <div class="need-controls">
                        <button class="need-btn" onclick="adjustNeed('${name}', -10)">-</button>
                        <button class="need-btn" onclick="adjustNeed('${name}', 10)">+</button>
                    </div>
                    <span class="need-value">${roundedValue}</span>
                </div>
                <div class="need-bar">
                    <div class="bar-fill ${barClass}" style="width: ${value}%"></div>
                    <div class="threshold-marker" style="left: ${CRITICAL_THRESHOLD}%"></div>
                </div>
            </div>
        `;
    }

    elements.needsList.innerHTML = html;
}

/**
 * Update behaviors panel with suggestions and dropdown
 */
function updateBehaviorsPanel(behaviors, availableBehaviors) {
    // Update suggestions list
    if (behaviors?.suggested) {
        let html = '';
        behaviors.suggested.forEach((b, i) => {
            const topClass = i === 0 ? 'top' : '';
            html += `
                <div class="behavior-item ${topClass}" title="Base: ${b.breakdown.base}, Need: ${b.breakdown.need}, Pers: ${b.breakdown.personality}, Opp: ${b.breakdown.opportunity}">
                    <span class="behavior-name">${b.name}</span>
                    <span class="behavior-score">${b.score.toFixed(3)}</span>
                </div>
            `;
        });
        elements.behaviorSuggestions.innerHTML = html;
    }

    // Update behavior dropdown (only if changed)
    if (availableBehaviors && elements.behaviorSelect.options.length !== availableBehaviors.length + 1) {
        let options = '<option value="">-- Select behavior --</option>';
        availableBehaviors.forEach(name => {
            options += `<option value="${name}">${name}</option>`;
        });
        elements.behaviorSelect.innerHTML = options;
    }
}

/**
 * Update triggers panel
 */
function updateTriggersPanel(context) {
    if (!context?.active_triggers) {
        elements.triggersList.innerHTML = '<span class="trigger-tag">No active triggers</span>';
        return;
    }

    const triggers = context.active_triggers;
    if (triggers.length === 0) {
        elements.triggersList.innerHTML = '<span class="trigger-tag">No active triggers</span>';
        return;
    }

    let html = '';
    triggers.forEach(t => {
        html += `<span class="trigger-tag active">${t}</span>`;
    });
    elements.triggersList.innerHTML = html;
}

/**
 * Update perception panel
 */
function updatePerceptionPanel(context) {
    if (!context) return;

    // Person detection
    const person = context.person || {};
    if (person.detected) {
        elements.personName.textContent = person.name || 'Unknown person';
        let details = [];
        if (person.familiarity !== undefined) {
            details.push(`Familiarity: ${Math.round(person.familiarity)}%`);
        }
        if (person.distance !== undefined && person.distance !== null) {
            details.push(`Distance: ${Math.round(person.distance)}cm`);
        }
        if (person.sentiment !== undefined) {
            details.push(`Sentiment: ${person.sentiment.toFixed(2)}`);
        }
        elements.personDetails.textContent = details.join(' | ');
    } else {
        elements.personName.textContent = 'No person detected';
        elements.personDetails.textContent = '';
    }

    // Environment conditions
    const env = context.environment || {};
    let envHtml = '';
    const envConditions = [
        { key: 'is_dark', label: 'Dark', danger: false },
        { key: 'is_loud', label: 'Loud', danger: false },
        { key: 'near_edge', label: 'Near Edge', danger: true },
    ];
    envConditions.forEach(c => {
        if (env[c.key]) {
            const cls = c.danger ? 'danger' : 'active';
            envHtml += `<span class="condition-tag ${cls}">${c.label}</span>`;
        }
    });
    elements.envConditions.innerHTML = envHtml || '<span class="condition-tag">Normal</span>';

    // Physical state
    const physical = context.physical || {};
    let physHtml = '';
    const physConditions = [
        { key: 'is_being_held', label: 'Held' },
        { key: 'is_being_petted', label: 'Petted' },
        { key: 'recent_bump', label: 'Bumped' },
    ];
    physConditions.forEach(c => {
        if (physical[c.key]) {
            physHtml += `<span class="condition-tag active">${c.label}</span>`;
        }
    });
    elements.physicalState.innerHTML = physHtml || '<span class="condition-tag">Normal</span>';
}

/**
 * Update behavior history panel
 */
function updateHistoryPanel(history) {
    if (!history || history.length === 0) {
        elements.historyList.innerHTML = '<div class="history-item">No history yet</div>';
        return;
    }

    let html = '';
    // Show newest first
    [...history].reverse().forEach(h => {
        const duration = h.duration ? `${h.duration.toFixed(1)}s` : '-';
        html += `
            <div class="history-item">
                <span class="history-name">${h.name}</span>
                <span class="history-status ${h.status}">${h.status}</span>
                <span class="history-duration">${duration}</span>
            </div>
        `;
    });
    elements.historyList.innerHTML = html;
}

/**
 * Update context panel (timing, spatial)
 */
function updateContextPanel(state) {
    // Timing info
    const timing = state.timing || {};
    const now = Date.now() / 1000;
    const perceptionAge = timing.last_perception ? (now - timing.last_perception).toFixed(1) : '-';
    const cognitionAge = timing.last_cognition ? (now - timing.last_cognition).toFixed(1) : '-';
    const executionAge = timing.last_execution ? (now - timing.last_execution).toFixed(1) : '-';

    elements.timingInfo.innerHTML = `
        Perception: ${perceptionAge}s ago<br>
        Cognition: ${cognitionAge}s ago<br>
        Execution: ${executionAge}s ago
    `;

    // Spatial info
    const spatial = state.world_context?.spatial || {};
    let spatialText = '';
    if (spatial.current_zone_type) {
        spatialText += `Zone: ${spatial.current_zone_type}`;
        if (spatial.current_zone_safety !== undefined) {
            spatialText += ` (safety: ${(spatial.current_zone_safety * 100).toFixed(0)}%)`;
        }
    }
    if (spatial.at_home_base) {
        spatialText += spatialText ? '<br>' : '';
        spatialText += 'At home base';
    }
    if (spatial.position_confidence !== undefined) {
        spatialText += spatialText ? '<br>' : '';
        spatialText += `Position confidence: ${(spatial.position_confidence * 100).toFixed(0)}%`;
    }
    elements.spatialInfo.innerHTML = spatialText || 'No spatial data';
}

/**
 * Adjust a need value via API
 */
async function adjustNeed(name, delta) {
    try {
        const response = await fetch('/api/control/need', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, delta }),
        });
        const result = await response.json();
        if (!result.success) {
            console.error('Failed to adjust need:', result.message);
        }
    } catch (e) {
        console.error('Error adjusting need:', e);
    }
}

/**
 * Trigger a specific behavior via API
 */
async function triggerBehavior() {
    const behaviorName = elements.behaviorSelect.value;
    if (!behaviorName) return;

    try {
        elements.triggerBtn.disabled = true;
        const response = await fetch('/api/control/trigger', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ behavior_name: behaviorName }),
        });
        const result = await response.json();
        if (!result.success) {
            console.error('Failed to trigger behavior:', result.message);
        }
    } catch (e) {
        console.error('Error triggering behavior:', e);
    } finally {
        elements.triggerBtn.disabled = false;
    }
}

/**
 * Toggle collapsible panels
 */
function setupCollapsibles() {
    document.querySelectorAll('.collapsible .panel-toggle').forEach(toggle => {
        toggle.addEventListener('click', () => {
            toggle.closest('.collapsible').classList.toggle('expanded');
        });
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupCollapsibles();
    connect();

    // Setup trigger button
    elements.triggerBtn.addEventListener('click', triggerBehavior);
});
