# Manual Integration Test Scenarios

## Prerequisites
1. Start server: `python -m server.main`
2. Start emulator: `cd /home/chris/myrobot && .venv/bin/python -m emulator`
3. Open browser to http://localhost:6080

## Scenario 1: Basic Connection

**Steps:**
1. Start only the emulator (server not running)
2. Observe connection status indicators

**Expected:**
- [ ] UI status shows "Connected" (green dot)
- [ ] Server status shows "Disconnected" (red dot)
- [ ] Raw state panel updates with emulator state

**Steps (continued):**
3. Start the server on port 6765
4. Wait 2-3 seconds

**Expected:**
- [ ] Server status changes to "Connected" (green dot)
- [ ] Console logs show "Virtual Pi connected to server"

---

## Scenario 2: Touch -> Sensor Data Flow

**Steps:**
1. With both server and emulator running
2. Click "Pet" button in emulator UI

**Expected:**
- [ ] Touch electrodes show "3, 4, 5, 6" in sensor panel
- [ ] Server logs show "SENSOR_DATA received" with touched electrodes
- [ ] Raw state shows `is_being_touched: true`

**Steps (continued):**
3. Click "Release" button

**Expected:**
- [ ] Touch electrodes show "None"
- [ ] `is_being_touched: false` in state

---

## Scenario 3: Pickup -> LOCAL_TRIGGER

**Steps:**
1. Click "Pick Up" button

**Expected:**
- [ ] IMU graph shows spike in Z-axis (blue line)
- [ ] Server logs show "LOCAL_TRIGGER received: picked_up_gentle"
- [ ] Raw state shows `simulated_pickup: true` briefly

---

## Scenario 4: Bump -> Visual Feedback

**Steps:**
1. Click "Bump" button

**Expected:**
- [ ] Robot avatar shakes visually (CSS animation)
- [ ] IMU graph shows spike in X-axis (red line)
- [ ] Server logs show "LOCAL_TRIGGER received: bump"

---

## Scenario 5: Shake -> Sustained Effect

**Steps:**
1. Click "Shake" button

**Expected:**
- [ ] Robot avatar shakes for ~1 second
- [ ] IMU graph shows sustained X/Y oscillation
- [ ] Server logs show "LOCAL_TRIGGER received: shake"

---

## Scenario 6: Motor Command Feedback

**Prerequisites:**
- Server must be running with a behavior that issues motor commands
- Or manually trigger via server API

**Steps:**
1. Wait for server to select a movement behavior (wander, explore)
2. OR use server debug interface to send motor command

**Expected:**
- [ ] Motor panel shows wheel bars updating
- [ ] Left/Right speed percentages change
- [ ] Direction indicator shows FORWARD/BACKWARD/LEFT/RIGHT
- [ ] Robot position updates in visualization
- [ ] Robot rotates based on heading changes

---

## Scenario 7: Expression Changes

**Prerequisites:**
- Server must send expression commands

**Steps:**
1. Trigger a behavior that changes expression (e.g., petting -> happy)
2. Click "Pet" button and wait for server to respond

**Expected:**
- [ ] Expression text in info panel changes (e.g., "happy")
- [ ] Robot face changes appearance (eyes shape, mouth)
- [ ] Happy: eyes become U-shaped, bigger smile
- [ ] Sad: inverted eyes, frown
- [ ] Surprised: large round eyes

---

## Scenario 8: Connection Recovery

**Steps:**
1. Note emulator shows both connections "Connected"
2. Stop the server (Ctrl+C in terminal)
3. Wait 2-3 seconds
4. Observe emulator behavior
5. Restart server

**Expected:**
- [ ] Emulator shows Server "Disconnected" within 5 seconds
- [ ] Emulator continues updating UI (sensor loop runs locally)
- [ ] After server restart, Server status returns to "Connected"
- [ ] Sensor streaming resumes automatically

---

## Scenario 9: IMU Graph Visualization

**Steps:**
1. Observe IMU graph while robot is idle
2. Click various simulation buttons and watch graph

**Expected:**
- [ ] Graph shows 3 colored lines: X (red), Y (teal), Z (blue)
- [ ] At rest, Z line is around -1.0 (gravity)
- [ ] X and Y lines hover near 0 with slight noise
- [ ] "Pick Up" shows Z spike toward 0
- [ ] "Bump" shows X spike
- [ ] "Shake" shows X/Y oscillation
- [ ] Graph scrolls smoothly (100 data points)

---

## Scenario 10: Full Behavior Loop

**Purpose:** Verify end-to-end cognition loop

**Steps:**
1. Start fresh with both server and emulator
2. Wait 30+ seconds without interaction (let needs decay)

**Expected (if loneliness behavior implemented):**
- [ ] Server's social need decreases over time
- [ ] Eventually "lonely" or "seek_attention" behavior activates
- [ ] Expression changes to "sad" or similar
- [ ] Robot may emit seeking motion (turn, scan)
- [ ] Sound may play

**Steps (continued):**
3. Click "Pet" button to simulate attention

**Expected:**
- [ ] Touch data reaches server
- [ ] Server recognizes petting
- [ ] Social need increases
- [ ] Expression changes to "happy"
- [ ] Behavior changes to happy response

---

## Debugging Tips

### Server Not Receiving Data
- Check server logs for WebSocket connection messages
- Verify port 6765 is not in use: `lsof -i :6765`
- Check emulator logs for "Failed to send" errors

### Expression Not Changing
- Verify server is connected (green dot)
- Check server logs for behavior selection
- May need to wait for cognition cycle (200ms)

### Motor Bars Not Moving
- Motor commands are only sent when behaviors require movement
- Check if any movement behavior is active in server

### Graph Not Updating
- Ensure IMU data is being sent (check raw state for imu_accel_* values)
- Graph only updates when connected and receiving state updates
