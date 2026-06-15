const levelBlueprints = [
  {
    number: 1,
    operation: "addition",
    numeralSystem: "decimal",
    leftMin: 1,
    leftMax: 9,
    rightMin: 1,
    rightMax: 9,
    minResult: 2,
    maxResult: 9,
    recentHistoryLimit: 18,
  },
  {
    number: 2,
    operation: "addition",
    numeralSystem: "decimal",
    leftMin: 1,
    leftMax: 9,
    rightMin: 1,
    rightMax: 9,
    minResult: 4,
    maxResult: 12,
    recentHistoryLimit: 22,
  },
  {
    number: 3,
    operation: "addition",
    numeralSystem: "decimal",
    leftMin: 2,
    leftMax: 9,
    rightMin: 2,
    rightMax: 9,
    minResult: 6,
    maxResult: 15,
    recentHistoryLimit: 24,
  },
  {
    number: 4,
    operation: "addition",
    numeralSystem: "decimal",
    leftMin: 3,
    leftMax: 12,
    rightMin: 3,
    rightMax: 12,
    minResult: 8,
    maxResult: 20,
    recentHistoryLimit: 28,
  },
  {
    number: 5,
    operation: "subtraction",
    numeralSystem: "decimal",
    leftMin: 6,
    leftMax: 15,
    rightMin: 1,
    rightMax: 9,
    minResult: 1,
    maxResult: 10,
    recentHistoryLimit: 22,
  },
  {
    number: 6,
    operation: "subtraction",
    numeralSystem: "decimal",
    leftMin: 8,
    leftMax: 19,
    rightMin: 2,
    rightMax: 10,
    minResult: 2,
    maxResult: 14,
    recentHistoryLimit: 24,
  },
  {
    number: 7,
    operation: "subtraction",
    numeralSystem: "decimal",
    leftMin: 10,
    leftMax: 24,
    rightMin: 3,
    rightMax: 12,
    minResult: 2,
    maxResult: 18,
    recentHistoryLimit: 28,
  },
  {
    number: 8,
    operation: "subtraction",
    numeralSystem: "decimal",
    leftMin: 14,
    leftMax: 31,
    rightMin: 4,
    rightMax: 15,
    minResult: 4,
    maxResult: 24,
    recentHistoryLimit: 30,
  },
  {
    number: 9,
    operation: "addition",
    numeralSystem: "binary",
    leftMin: 2,
    leftMax: 12,
    rightMin: 1,
    rightMax: 10,
    minResult: 3,
    maxResult: 20,
    recentHistoryLimit: 24,
  },
  {
    number: 10,
    operation: "subtraction",
    numeralSystem: "binary",
    leftMin: 8,
    leftMax: 20,
    rightMin: 1,
    rightMax: 12,
    minResult: 1,
    maxResult: 16,
    recentHistoryLimit: 24,
  },
];

const state = {
  soundEnabled: true,
  musicEnabled: true,
  view: "menu",
  currentLevelIndex: 0,
  currentLevelData: null,
  levelLocked: false,
  animationRunId: 0,
  problemHistoryByLevel: {},
};

class AudioEngine {
  constructor() {
    this.context = null;
    this.masterGain = null;
    this.soundGain = null;
    this.musicGain = null;
    this.musicLoopHandle = null;
    this.noiseBuffer = null;
  }

  hasWebAudioSupport() {
    return Boolean(window.AudioContext || window.webkitAudioContext);
  }

  async ensureStarted() {
    if (!this.hasWebAudioSupport()) {
      return;
    }

    if (!this.context) {
      this.createContext();
    }

    if (this.context.state === "suspended") {
      await this.context.resume();
    }

    this.syncWithState();
  }

  createContext() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;

    if (!AudioContextClass) {
      return;
    }

    this.context = new AudioContextClass();
    this.masterGain = this.context.createGain();
    this.soundGain = this.context.createGain();
    this.musicGain = this.context.createGain();

    this.masterGain.gain.value = 0.9;
    this.soundGain.gain.value = 0;
    this.musicGain.gain.value = 0;

    this.soundGain.connect(this.masterGain);
    this.musicGain.connect(this.masterGain);
    this.masterGain.connect(this.context.destination);
    this.noiseBuffer = this.createNoiseBuffer();
  }

  createNoiseBuffer() {
    const buffer = this.context.createBuffer(1, this.context.sampleRate * 1.5, this.context.sampleRate);
    const data = buffer.getChannelData(0);

    for (let index = 0; index < data.length; index += 1) {
      data[index] = Math.random() * 2 - 1;
    }

    return buffer;
  }

  syncWithState() {
    if (!this.context) {
      return;
    }

    const now = this.context.currentTime;

    this.soundGain.gain.cancelScheduledValues(now);
    this.soundGain.gain.linearRampToValueAtTime(state.soundEnabled ? 0.22 : 0, now + 0.05);

    this.musicGain.gain.cancelScheduledValues(now);
    this.musicGain.gain.linearRampToValueAtTime(state.musicEnabled ? 0.12 : 0, now + 0.12);

    if (state.musicEnabled) {
      this.startMusicLoop();
    } else {
      this.stopMusicLoop();
    }
  }

  stopMusicLoop() {
    if (this.musicLoopHandle) {
      window.clearInterval(this.musicLoopHandle);
      this.musicLoopHandle = null;
    }
  }

  startMusicLoop() {
    if (!this.context || this.musicLoopHandle) {
      return;
    }

    this.playMusicPhrase();
    this.musicLoopHandle = window.setInterval(() => {
      if (!this.context || this.context.state !== "running" || !state.musicEnabled) {
        return;
      }

      this.playMusicPhrase();
    }, 3200);
  }

  canPlaySound() {
    return Boolean(this.context && state.soundEnabled);
  }

  canPlayMusic() {
    return Boolean(this.context && state.musicEnabled);
  }

  playTone({
    start,
    duration,
    frequency,
    targetFrequency = frequency,
    gain = 0.05,
    type = "sine",
    destination = this.soundGain,
  }) {
    if (!this.context || !destination) {
      return;
    }

    const oscillator = this.context.createOscillator();
    const gainNode = this.context.createGain();

    oscillator.type = type;
    oscillator.frequency.setValueAtTime(frequency, start);
    oscillator.frequency.exponentialRampToValueAtTime(Math.max(20, targetFrequency), start + duration);

    gainNode.gain.setValueAtTime(0.0001, start);
    gainNode.gain.exponentialRampToValueAtTime(gain, start + Math.min(0.03, duration * 0.4));
    gainNode.gain.exponentialRampToValueAtTime(0.0001, start + duration);

    oscillator.connect(gainNode);
    gainNode.connect(destination);
    oscillator.start(start);
    oscillator.stop(start + duration + 0.02);
  }

  playNoise({ start, duration, gain = 0.028, lowpassStart = 1700, lowpassEnd = 260 }) {
    if (!this.context || !this.noiseBuffer) {
      return;
    }

    const source = this.context.createBufferSource();
    const filter = this.context.createBiquadFilter();
    const gainNode = this.context.createGain();

    source.buffer = this.noiseBuffer;
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(lowpassStart, start);
    filter.frequency.exponentialRampToValueAtTime(Math.max(50, lowpassEnd), start + duration);

    gainNode.gain.setValueAtTime(0.0001, start);
    gainNode.gain.exponentialRampToValueAtTime(gain, start + 0.03);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, start + duration);

    source.connect(filter);
    filter.connect(gainNode);
    gainNode.connect(this.soundGain);
    source.start(start);
    source.stop(start + duration + 0.02);
  }

  playUIClick() {
    if (!this.canPlaySound()) {
      return;
    }

    const now = this.context.currentTime;
    this.playTone({
      start: now,
      duration: 0.05,
      frequency: 760,
      targetFrequency: 610,
      gain: 0.04,
      type: "square",
    });
    this.playTone({
      start: now + 0.04,
      duration: 0.045,
      frequency: 960,
      targetFrequency: 790,
      gain: 0.03,
      type: "square",
    });
  }

  playJump() {
    if (!this.canPlaySound()) {
      return;
    }

    const now = this.context.currentTime;
    this.playTone({
      start: now,
      duration: 0.18,
      frequency: 240,
      targetFrequency: 430,
      gain: 0.05,
      type: "triangle",
    });
    this.playTone({
      start: now + 0.08,
      duration: 0.16,
      frequency: 440,
      targetFrequency: 320,
      gain: 0.035,
      type: "square",
    });
  }

  playFall() {
    if (!this.canPlaySound()) {
      return;
    }

    const now = this.context.currentTime;
    this.playTone({
      start: now,
      duration: 0.42,
      frequency: 310,
      targetFrequency: 62,
      gain: 0.04,
      type: "sawtooth",
    });
    this.playNoise({
      start: now,
      duration: 0.34,
      gain: 0.025,
      lowpassStart: 1200,
      lowpassEnd: 180,
    });
  }

  playCorrect() {
    if (!this.canPlaySound()) {
      return;
    }

    const now = this.context.currentTime;
    const notes = [392, 494, 587];

    notes.forEach((frequency, index) => {
      this.playTone({
        start: now + index * 0.09,
        duration: 0.22,
        frequency,
        targetFrequency: frequency * 1.02,
        gain: 0.045,
        type: "triangle",
      });
    });
  }

  playWrong() {
    if (!this.canPlaySound()) {
      return;
    }

    const now = this.context.currentTime;
    this.playTone({
      start: now,
      duration: 0.2,
      frequency: 220,
      targetFrequency: 120,
      gain: 0.06,
      type: "sawtooth",
    });
    this.playTone({
      start: now + 0.09,
      duration: 0.26,
      frequency: 150,
      targetFrequency: 70,
      gain: 0.055,
      type: "square",
    });
  }

  playLevelStart(levelNumber) {
    if (!this.canPlaySound()) {
      return;
    }

    const now = this.context.currentTime;
    const base = 196 + levelNumber * 8;

    this.playTone({
      start: now,
      duration: 0.18,
      frequency: base,
      targetFrequency: base * 1.12,
      gain: 0.035,
      type: "triangle",
    });
    this.playTone({
      start: now + 0.12,
      duration: 0.2,
      frequency: base * 1.25,
      targetFrequency: base * 1.32,
      gain: 0.03,
      type: "sine",
    });
  }

  playVictory() {
    if (!this.canPlaySound()) {
      return;
    }

    const now = this.context.currentTime;
    const notes = [392, 494, 587, 784];

    notes.forEach((frequency, index) => {
      this.playTone({
        start: now + index * 0.12,
        duration: 0.28,
        frequency,
        targetFrequency: frequency * 1.01,
        gain: 0.05,
        type: "triangle",
      });
    });
  }

  playMusicPhrase() {
    if (!this.canPlayMusic()) {
      return;
    }

    const start = this.context.currentTime + 0.04;
    const bassNotes = [82.41, 92.5, 73.42, 61.74];
    const leadNotes = [164.81, 196, 174.61, 146.83];

    bassNotes.forEach((frequency, index) => {
      this.playTone({
        start: start + index * 0.78,
        duration: 0.62,
        frequency,
        targetFrequency: frequency * 0.98,
        gain: 0.022,
        type: "sine",
        destination: this.musicGain,
      });
    });

    leadNotes.forEach((frequency, index) => {
      this.playTone({
        start: start + index * 0.78 + 0.24,
        duration: 0.26,
        frequency,
        targetFrequency: frequency * 1.01,
        gain: 0.012,
        type: "triangle",
        destination: this.musicGain,
      });
    });
  }
}

const audio = new AudioEngine();

const startButton = document.querySelector("#startButton");
const soundButton = document.querySelector("#soundButton");
const musicButton = document.querySelector("#musicButton");
const statusText = document.querySelector("#statusText");
const menuStage = document.querySelector(".menu-stage");
const gameStage = document.querySelector("#gameStage");
const dustField = document.querySelector("#dustField");
const problemText = document.querySelector("#problemText");
const levelTitle = document.querySelector("#levelTitle");
const levelMessage = document.querySelector("#levelMessage");
const chamberPanel = document.querySelector("#chamberPanel");
const hero = document.querySelector("#hero");
const backToMenuButton = document.querySelector("#backToMenuButton");
const outcomeOverlay = document.querySelector("#outcomeOverlay");
const outcomeCard = document.querySelector("#outcomeCard");
const outcomeLabel = document.querySelector("#outcomeLabel");
const outcomeTitle = document.querySelector("#outcomeTitle");
const outcomeMessage = document.querySelector("#outcomeMessage");
const outcomePrimaryButton = document.querySelector("#outcomePrimaryButton");
const outcomeMenuButton = document.querySelector("#outcomeMenuButton");
const wellButtons = [...document.querySelectorAll(".well-choice")];

function setStatus(message) {
  statusText.textContent = message;
}

function syncToggleButton(button, label, enabled) {
  button.setAttribute("aria-pressed", String(enabled));
  button.textContent = `${label}: ${enabled ? "On" : "Off"}`;
}

function toggleSetting(key, button, label, enabledMessage, disabledMessage) {
  state[key] = !state[key];
  syncToggleButton(button, label, state[key]);
  audio.syncWithState();
  setStatus(state[key] ? enabledMessage : disabledMessage);
}

function createDustParticles() {
  const particleCount = 24;

  for (let index = 0; index < particleCount; index += 1) {
    const particle = document.createElement("span");
    particle.className = "dust-particle";
    particle.style.setProperty("--size", `${Math.random() * 4 + 2}px`);
    particle.style.setProperty("--duration", `${Math.random() * 10 + 12}s`);
    particle.style.setProperty("--delay", `${Math.random() * -18}s`);
    particle.style.setProperty("--drift", `${Math.random() * 6 - 3}rem`);
    particle.style.left = `${Math.random() * 100}%`;
    particle.style.top = `${Math.random() * 100}%`;
    dustField.appendChild(particle);
  }
}

function shuffleArray(items) {
  const next = [...items];

  for (let index = next.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [next[index], next[swapIndex]] = [next[swapIndex], next[index]];
  }

  return next;
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function formatValue(value, numeralSystem) {
  return numeralSystem === "binary" ? value.toString(2) : String(value);
}

function getCurrentBlueprint() {
  return levelBlueprints[state.currentLevelIndex];
}

function buildProblemText(left, operation, right, numeralSystem) {
  const leftLabel = formatValue(left, numeralSystem);
  const rightLabel = formatValue(right, numeralSystem);
  const operatorLabel = operation === "addition" ? "+" : "-";
  return `${leftLabel} ${operatorLabel} ${rightLabel} = ?`;
}

function buildProblemSignature(left, operation, right, numeralSystem) {
  return `${numeralSystem}:${left}:${operation}:${right}`;
}

function getProblemHistory(levelNumber) {
  return state.problemHistoryByLevel[levelNumber] ?? [];
}

function rememberProblemSignature(levelNumber, signature, limit) {
  const history = [...getProblemHistory(levelNumber), signature];

  while (history.length > limit) {
    history.shift();
  }

  state.problemHistoryByLevel[levelNumber] = history;
}

function generateWrongAnswerValues(correctValue, minValue, maxValue, left, right, operation) {
  const candidates = new Set([
    correctValue - 3,
    correctValue - 2,
    correctValue - 1,
    correctValue + 1,
    correctValue + 2,
    correctValue + 3,
    correctValue + 4,
    left,
    right,
    left + right,
    Math.abs(left - right),
  ]);

  if (operation === "subtraction") {
    candidates.add(correctValue + right);
    candidates.add(left - right - 1);
    candidates.add(left - right + 1);
  }

  if (operation === "addition") {
    candidates.add(correctValue + left);
    candidates.add(correctValue - right);
  }

  const validCandidates = shuffleArray(
    [...candidates].filter(
      (value) => value >= minValue && value <= maxValue && value !== correctValue,
    ),
  );

  const wrongValues = validCandidates.slice(0, 2);
  let paddingValue = Math.max(correctValue + 5, minValue + 1);

  while (wrongValues.length < 2) {
    if (!wrongValues.includes(paddingValue) && paddingValue !== correctValue) {
      wrongValues.push(paddingValue);
    }
    paddingValue += 1;
  }

  return wrongValues;
}

function generateOperands(blueprint) {
  const history = getProblemHistory(blueprint.number);
  const minResult = blueprint.minResult ?? 1;
  const maxResult = blueprint.maxResult ?? Number.POSITIVE_INFINITY;
  const recentHistoryLimit = blueprint.recentHistoryLimit ?? 20;
  let attempt = 0;

  while (attempt < 120) {
    const left = randomInt(blueprint.leftMin, blueprint.leftMax);
    const right = randomInt(blueprint.rightMin, blueprint.rightMax);
    const correctValue = blueprint.operation === "addition" ? left + right : left - right;
    const signature = buildProblemSignature(left, blueprint.operation, right, blueprint.numeralSystem);

    if (
      correctValue >= minResult &&
      correctValue <= maxResult &&
      !history.includes(signature)
    ) {
      rememberProblemSignature(blueprint.number, signature, recentHistoryLimit);
      return { left, right, correctValue };
    }

    attempt += 1;
  }

  const fallbackLeft = blueprint.operation === "subtraction"
    ? Math.max(blueprint.leftMin, minResult + blueprint.rightMin)
    : blueprint.leftMin;
  const fallbackRight = blueprint.rightMin;
  const fallbackCorrect = blueprint.operation === "addition"
    ? fallbackLeft + fallbackRight
    : fallbackLeft - fallbackRight;

  rememberProblemSignature(
    blueprint.number,
    buildProblemSignature(
      fallbackLeft,
      blueprint.operation,
      fallbackRight,
      blueprint.numeralSystem,
    ),
    recentHistoryLimit,
  );

  return {
    left: fallbackLeft,
    right: fallbackRight,
    correctValue: fallbackCorrect,
  };
}

function buildLevelData(blueprint) {
  const { left, right, correctValue } = generateOperands(blueprint);
  const wrongValues = generateWrongAnswerValues(
    correctValue,
    blueprint.minResult ?? 1,
    blueprint.maxResult ?? correctValue + 10,
    left,
    right,
    blueprint.operation,
  );
  const answers = [
    {
      value: correctValue,
      label: formatValue(correctValue, blueprint.numeralSystem),
    },
    ...wrongValues.map((value) => ({
      value,
      label: formatValue(value, blueprint.numeralSystem),
    })),
  ];

  return {
    number: blueprint.number,
    operation: blueprint.operation,
    numeralSystem: blueprint.numeralSystem,
    prompt: buildProblemText(left, blueprint.operation, right, blueprint.numeralSystem),
    correctValue,
    answers,
  };
}

function resetHero() {
  hero.classList.remove("is-jumping", "is-dropping", "is-defeated");
  hero.style.setProperty("--hero-shift", "0px");
}

function setView(view) {
  state.view = view;
  const showMenu = view === "menu";
  menuStage.hidden = !showMenu;
  gameStage.hidden = showMenu;
}

function moveHeroToWell(button) {
  const heroRect = hero.getBoundingClientRect();
  const buttonRect = button.getBoundingClientRect();
  const heroCenter = heroRect.left + heroRect.width / 2;
  const buttonCenter = buttonRect.left + buttonRect.width / 2;
  const shift = buttonCenter - heroCenter;
  hero.style.setProperty("--hero-shift", `${shift}px`);
}

function clearWellStates() {
  for (const button of wellButtons) {
    button.disabled = false;
    button.classList.remove("is-correct", "is-wrong");
  }
}

function hideOutcomeOverlay() {
  outcomeOverlay.hidden = true;
  outcomeCard.classList.remove("is-defeat", "is-victory");
}

function showOutcomeOverlay({
  mode,
  label,
  title,
  message,
  primaryLabel,
  primaryAction,
}) {
  outcomeOverlay.hidden = false;
  outcomeCard.classList.remove("is-defeat", "is-victory");
  outcomeCard.classList.add(mode === "victory" ? "is-victory" : "is-defeat");
  outcomeLabel.textContent = label;
  outcomeTitle.textContent = title;
  outcomeMessage.textContent = message;
  outcomePrimaryButton.textContent = primaryLabel;
  outcomePrimaryButton.dataset.action = primaryAction;
}

function renderCurrentLevel() {
  state.animationRunId += 1;
  state.levelLocked = false;
  state.currentLevelData = buildLevelData(getCurrentBlueprint());
  chamberPanel.classList.remove("is-shaking");
  resetHero();
  clearWellStates();
  hideOutcomeOverlay();

  levelTitle.textContent = `Level ${state.currentLevelData.number}`;
  problemText.textContent = state.currentLevelData.prompt;
  levelMessage.textContent = `Read the wall. Choose the one correct answer for Level ${state.currentLevelData.number}.`;

  const answers = shuffleArray(state.currentLevelData.answers);

  answers.forEach((answer, index) => {
    const button = wellButtons[index];
    const sign = button.querySelector(".well-sign");
    button.dataset.answerValue = String(answer.value);
    sign.textContent = answer.label;
  });
}

function openLevelSequence() {
  menuStage.classList.remove("is-descending");
  window.requestAnimationFrame(() => {
    menuStage.classList.add("is-descending");
  });

  state.currentLevelIndex = 0;
  setView("game");
  renderCurrentLevel();
  audio.playLevelStart(state.currentLevelData.number);

  const soundState = state.soundEnabled ? "Torch crackle follows the miner." : "The chamber is unnervingly silent.";
  const musicState = state.musicEnabled ? "The dungeon rhythm continues below." : "Only the lava watches.";
  setStatus(`Level ${state.currentLevelData.number} has begun. ${soundState} ${musicState}`);
}

function openNextLevel() {
  if (state.currentLevelIndex >= levelBlueprints.length - 1) {
    return;
  }

  state.currentLevelIndex += 1;
  renderCurrentLevel();
  audio.playLevelStart(state.currentLevelData.number);
  setStatus(`Level ${state.currentLevelData.number} has begun. A new wall problem awaits.`);
}

function completeLevel() {
  const level = state.currentLevelData;
  const currentRun = state.animationRunId;
  const isLastLevel = state.currentLevelIndex === levelBlueprints.length - 1;

  if (isLastLevel) {
    levelMessage.textContent = `Correct. Level ${level.number} is cleared. All 10 levels of the current run are complete.`;
    audio.playVictory();
    showOutcomeOverlay({
      mode: "victory",
      label: "Run Complete",
      title: "Victory",
      message: "You solved all 10 levels and survived the mine. Clear thinking carried the miner through every shaft.",
      primaryLabel: "Run Again",
      primaryAction: "restart-run",
    });
    setStatus(`Level ${level.number} complete. The full 10-level run is finished.`);
    return;
  }

  audio.playCorrect();
  const nextLevelNumber = levelBlueprints[state.currentLevelIndex + 1].number;
  levelMessage.textContent = `Correct. The miner is pulled deeper. Level ${nextLevelNumber} begins now.`;
  setStatus(`Level ${level.number} complete. Descending automatically to Level ${nextLevelNumber}.`);

  window.setTimeout(() => {
    if (state.animationRunId !== currentRun || state.view !== "game") {
      return;
    }

    openNextLevel();
  }, 1200);
}

function failLevel() {
  const level = state.currentLevelData;
  levelMessage.textContent = `Wrong answer. The miner crashes into the dark. Restart Level ${level.number}.`;
  audio.playWrong();
  showOutcomeOverlay({
    mode: "defeat",
    label: "Run Failed",
    title: "Game Over",
    message: `The miner chose the wrong tunnel on Level ${level.number}. The run is lost, but a new descent can begin from Level 1.`,
    primaryLabel: "Restart Run",
    primaryAction: "restart-run",
  });
  setStatus(`The mine rejects the careless. Level ${level.number} can be attempted again with a new problem.`);
}

function resolveChoice(button) {
  if (state.levelLocked) {
    return;
  }

  const runId = state.animationRunId + 1;
  state.animationRunId = runId;
  state.levelLocked = true;
  wellButtons.forEach((choice) => {
    choice.disabled = true;
  });

  const chosenAnswer = Number(button.dataset.answerValue);
  const isCorrect = chosenAnswer === state.currentLevelData.correctValue;

  button.classList.add(isCorrect ? "is-correct" : "is-wrong");
  moveHeroToWell(button);
  hero.classList.remove("is-jumping", "is-dropping");
  void hero.offsetWidth;

  window.requestAnimationFrame(() => {
    if (state.animationRunId !== runId) {
      return;
    }

    audio.playJump();
    hero.classList.add("is-jumping");
  });

  window.setTimeout(() => {
    if (state.animationRunId !== runId) {
      return;
    }

    hero.classList.remove("is-jumping");

    if (!isCorrect) {
      chamberPanel.classList.add("is-shaking");
      hero.classList.add("is-defeated");
    }

    void hero.offsetWidth;
    audio.playFall();
    hero.classList.add("is-dropping");
  }, 700);

  window.setTimeout(() => {
    if (state.animationRunId !== runId) {
      return;
    }

    if (isCorrect) {
      completeLevel();
    } else {
      failLevel();
    }
  }, 1320);
}

function returnToMenu() {
  state.animationRunId += 1;
  setView("menu");
  resetHero();
  chamberPanel.classList.remove("is-shaking");
  hideOutcomeOverlay();
  setStatus("The lava glows below. Level 1 is waiting with a new problem.");
}

function handleGameKey(event) {
  if (state.view !== "game" || state.levelLocked) {
    return;
  }

  const index = Number(event.key) - 1;

  if (index >= 0 && index < wellButtons.length) {
    resolveChoice(wellButtons[index]);
  }
}

syncToggleButton(soundButton, "Sound", state.soundEnabled);
syncToggleButton(musicButton, "Music", state.musicEnabled);
createDustParticles();
hideOutcomeOverlay();

startButton.addEventListener("click", async () => {
  await audio.ensureStarted();
  audio.playUIClick();
  openLevelSequence();
});

backToMenuButton.addEventListener("click", async () => {
  await audio.ensureStarted();
  audio.playUIClick();
  returnToMenu();
});

outcomePrimaryButton.addEventListener("click", async () => {
  await audio.ensureStarted();
  audio.playUIClick();

  if (outcomePrimaryButton.dataset.action === "restart-run") {
    openLevelSequence();
  }
});

outcomeMenuButton.addEventListener("click", async () => {
  await audio.ensureStarted();
  audio.playUIClick();
  returnToMenu();
});

wellButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    await audio.ensureStarted();
    resolveChoice(button);
  });
});

soundButton.addEventListener("click", async () => {
  await audio.ensureStarted();

  if (state.soundEnabled) {
    audio.playUIClick();
  }

  toggleSetting(
    "soundEnabled",
    soundButton,
    "Sound",
    "Torch crackle, jumps, and tunnel echoes will be heard.",
    "Sound effects are muted. The cave feels heavier now.",
  );
});

musicButton.addEventListener("click", async () => {
  await audio.ensureStarted();

  if (state.soundEnabled) {
    audio.playUIClick();
  }

  toggleSetting(
    "musicEnabled",
    musicButton,
    "Music",
    "The dark mine melody returns.",
    "Music is muted. The lava glows in uneasy silence.",
  );
});

document.addEventListener("keydown", handleGameKey);
