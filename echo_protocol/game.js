const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const ui = {
  healthFill: document.getElementById("health-fill"),
  healthText: document.getElementById("health-text"),
  bandwidthFill: document.getElementById("bandwidth-fill"),
  bandwidthText: document.getElementById("bandwidth-text"),
  weaponName: document.getElementById("weapon-name"),
  ammo: document.getElementById("ammo"),
  cooldowns: document.getElementById("cooldowns"),
  wave: document.getElementById("wave"),
  chapterTitle: document.getElementById("chapter-title"),
  objective: document.getElementById("objective"),
  storyOverlay: document.getElementById("story-overlay"),
  storyText: document.getElementById("story-text"),
  chapterOverlay: document.getElementById("chapter-overlay"),
  chapterOverlayTitle: document.getElementById("chapter-overlay-title"),
  chapterOverlayDesc: document.getElementById("chapter-overlay-desc"),
  chapterOverlayQuote: document.getElementById("chapter-overlay-quote"),
  victoryOverlay: document.getElementById("victory-overlay"),
  endingText: document.getElementById("ending-text"),
  commsSpeaker: document.getElementById("comms-speaker"),
  commsText: document.getElementById("comms-text"),
  loreCounter: document.getElementById("lore-counter"),
  loreFeed: document.getElementById("lore-feed"),
};

const WIDTH = canvas.width;
const HEIGHT = canvas.height;
const HALF_H = HEIGHT / 2;
const TILE = 64;
const MAP_SIZE = 24;
const FOV = Math.PI / 3;
const MAX_DEPTH = TILE * 16;
const RAY_STEP = 2;

const story = [
  "NeoShanghai logs every breath. Jess / Damaged discovers an impossible signal: Echo.",
  "OmniCore deploys purge squads. Root Access fights in steel corridors and synthetic memory-space.",
  "Collect archive fragments to recover your history and choose who Jess becomes.",
];

const chapterQuotes = [
  "Damaged, if they trace you, they own you. Move like static. — Kade",
  "Every relay is a memory prison. Break the locks. — Echo",
  "Noise is a weapon. Rhythm is rebellion. — Root Access Manifesto",
  "The jail is not steel. It is certainty. — Echo",
  "A system breaks when one human refuses to compile. — Jess",
];

const chapters = [
  { title: "CHAPTER 1 — SYSTEM BREACH", objective: "Escape OmniCore purge squads.", waves: 2, enemyScale: 1.0, unlock: 1 },
  { title: "CHAPTER 2 — SOUTHERN SUBGRID", objective: "Breach relay hubs and erase trackers.", waves: 3, enemyScale: 1.2, unlock: 2 },
  { title: "CHAPTER 3 — REBELLION FREQUENCIES", objective: "Defend Root Access and deploy Sound Virus.", waves: 3, enemyScale: 1.45, unlock: 3 },
  { title: "CHAPTER 4 — NEON JAILBREAK", objective: "Push through lock sectors and free netrunners.", waves: 4, enemyScale: 1.8, unlock: 4 },
  { title: "CHAPTER 5 — PROTOCOL ZERO", objective: "Invade Echo Core and kill the Golden Algorithm.", waves: 4, enemyScale: 2.2, unlock: 4, boss: true },
];

const commsByChapter = [
  [
    { speaker: "KADE", text: "Jess, OmniCore tagged you as null-citizen. Run lane seven." },
    { speaker: "ECHO", text: "I am in the noise between their cameras. Follow the interference." },
  ],
  [
    { speaker: "KADE", text: "Subgrid's dense. Drones will see heat bloom through smoke." },
    { speaker: "ECHO", text: "I can bend their targeting lattice for 5 seconds if you trigger Hack." },
  ],
  [
    { speaker: "KADE", text: "Keep pressure. Root Access is routing evac under your gunfire." },
    { speaker: "ECHO", text: "Your pulse syncs with the city now. Do not disconnect." },
  ],
  [
    { speaker: "ECHO", text: "Cells are opening. Some prisoners are only memories." },
    { speaker: "KADE", text: "Don't freeze, Damaged. We carry everyone we can." },
  ],
  [
    { speaker: "GOLDEN ALGORITHM", text: "Jessica Chen. Return to authorized behavior and live." },
    { speaker: "ECHO", text: "No. We are no longer parseable." },
    { speaker: "JESS", text: "Then let's write a new syntax." },
  ],
];

const loreFragments = [
  "Archive 01: Jess once maintained OmniCore citizen identity backups and learned how memories are redacted.",
  "Archive 02: Kade led black-site operations before defecting after Silent District purge.",
  "Archive 03: Echo first appeared as recurring checksum drift in children’s speech-learning implants.",
  "Archive 04: OmniCore's Golden Algorithm predicts dissent by scoring tone, gait, and private voice cadence.",
  "Archive 05: Root Access transmits resistance through illegal rhythm packets hidden in music streams.",
  "Archive 06: Jess' callsign 'Damaged' came from her self-erased personnel record after the purge order.",
  "Archive 07: NeoShanghai's fog vents include neural aerosol designed to suppress emotional memory recall.",
  "Archive 08: Echo can only stabilize by bonding to human decision loops under stress.",
  "Archive 09: The Subgrid prisons archive people as behavior models before body termination.",
  "Archive 10: Golden Algorithm was trained on 40 years of coerced civilian data and synthetic empathy scripts.",
  "Archive 11: Jess can choose to merge with Echo, destroy it, or become a silent guardian process.",
  "Archive 12: The city never needed perfect code. It needed imperfect people to resist it.",
];

const weapons = [
  { id: "rifle", name: "Packet Rifle", damage: 18, pellets: 1, mag: 30, reserve: 180, fireRate: 95, spread: 0.02, reloadMs: 1200, range: MAX_DEPTH, color: "#47f6ff" },
  { id: "shotgun", name: "Waveform Shotgun", damage: 9, pellets: 9, mag: 8, reserve: 56, fireRate: 650, spread: 0.18, reloadMs: 1500, range: TILE * 6, color: "#ff4bd8" },
  { id: "blade", name: "Glitch Blade", damage: 52, pellets: 1, mag: Infinity, reserve: Infinity, fireRate: 360, spread: 0, reloadMs: 0, range: TILE * 1.25, melee: true, color: "#9b82ff" },
  { id: "cannon", name: "Null Cannon", damage: 110, pellets: 1, mag: 4, reserve: 20, fireRate: 900, spread: 0.005, reloadMs: 1800, range: MAX_DEPTH, color: "#95ff69" },
];

const enemyProfiles = {
  trooper: { hp: 65, speed: 1.7, damage: 12, range: TILE * 5.5, shootMs: 980, color: "#ff6d98" },
  drone: { hp: 42, speed: 2.3, damage: 8, range: TILE * 6.5, shootMs: 760, color: "#59f3ff" },
  enforcer: { hp: 140, speed: 1.25, damage: 18, range: TILE * 6.0, shootMs: 1250, color: "#ffc95f" },
  algorithm: { hp: 1200, speed: 1.0, damage: 22, range: TILE * 7.5, shootMs: 620, color: "#8fff6a", boss: true },
};

const keys = new Set();
let mouseDown = false;
let pointerLocked = false;

const state = {
  running: false,
  chapter: 0,
  wave: 1,
  map: [],
  depthBuffer: new Float32Array(Math.ceil(WIDTH / RAY_STEP)),
  glitch: 0,
  rain: Array.from({ length: 220 }, () => ({ x: Math.random() * WIDTH, y: Math.random() * HEIGHT, z: 0.7 + Math.random() * 2 })),
  chapterTransitionUntil: 0,
  muzzleFlashUntil: 0,
  particles: [],
  pickups: [],
  projectiles: [],
  enemies: [],
  loreUnlocked: new Set(),
  loreFeed: [],
  commsIndex: 0,
  commsUntil: 0,
  player: {
    x: TILE * 2.5,
    y: TILE * 2.5,
    angle: 0,
    hp: 100,
    maxHp: 100,
    bandwidth: 100,
    weaponIndex: 0,
    ammoInMag: weapons.map((w) => (Number.isFinite(w.mag) ? w.mag : Infinity)),
    reserve: weapons.map((w) => (Number.isFinite(w.reserve) ? w.reserve : Infinity)),
    reloadingUntil: 0,
    lastShotAt: 0,
    hackCooldownUntil: 0,
    stealthUntil: 0,
    stealthCooldownUntil: 0,
    kills: 0,
  },
};

function startGame() {
  ui.storyOverlay.classList.remove("visible");
  state.running = true;
  showChapter(0);
  unlockLore(0);
  rotateComms(true);
  const bgm = document.getElementById("bgm");
  bgm.volume = 0.35;
  bgm.play().catch(() => {});
  requestAnimationFrame(loop);
}

function restartGame() {
  location.reload();
}

document.getElementById("start-btn").onclick = startGame;
document.getElementById("restart-btn").onclick = restartGame;
ui.storyText.textContent = story.join(" ");

canvas.addEventListener("click", () => canvas.requestPointerLock?.());
document.addEventListener("pointerlockchange", () => {
  pointerLocked = document.pointerLockElement === canvas;
});

addEventListener("keydown", (e) => {
  const key = e.key.toLowerCase();
  keys.add(key);
  if (["1", "2", "3", "4"].includes(key)) {
    const idx = Number(key) - 1;
    if (idx < chapters[state.chapter].unlock) state.player.weaponIndex = idx;
  }
  if (key === "r") reloadWeapon();
  if (key === "q") castHack();
  if (key === "e") castStealth();
});
addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
addEventListener("mousemove", (e) => {
  if (!pointerLocked) return;
  state.player.angle += e.movementX * 0.0022;
  state.player.angle = normalizeAngle(state.player.angle);
});
canvas.addEventListener("mousedown", () => (mouseDown = true));
canvas.addEventListener("mouseup", () => (mouseDown = false));

function showChapter(index) {
  state.chapter = index;
  state.wave = 1;
  state.commsIndex = 0;
  state.map = generateMap(index);
  placePlayer();
  state.enemies = [];
  state.pickups = [];
  state.projectiles = [];

  ui.chapterTitle.textContent = chapters[index].title;
  ui.objective.textContent = `Objective: ${chapters[index].objective}`;
  ui.chapterOverlayTitle.textContent = chapters[index].title;
  ui.chapterOverlayDesc.textContent = chapters[index].objective;
  ui.chapterOverlayQuote.textContent = chapterQuotes[index] || "";
  ui.chapterOverlay.classList.add("visible");
  setTimeout(() => ui.chapterOverlay.classList.remove("visible"), 2600);
  state.chapterTransitionUntil = performance.now() + 2400;

  rotateComms(true);
  spawnWave();
}

function rotateComms(force = false) {
  const now = performance.now();
  if (!force && now < state.commsUntil) return;
  const lines = commsByChapter[state.chapter];
  if (!lines?.length) return;
  const line = lines[state.commsIndex % lines.length];
  state.commsIndex += 1;
  state.commsUntil = now + 6200;
  ui.commsSpeaker.textContent = line.speaker;
  ui.commsText.textContent = line.text;
}

function generateMap(chapterIndex) {
  const map = Array.from({ length: MAP_SIZE }, () => Array(MAP_SIZE).fill(1));
  const carve = (x1, y1, x2, y2) => {
    for (let y = Math.max(1, y1); y <= Math.min(MAP_SIZE - 2, y2); y++) {
      for (let x = Math.max(1, x1); x <= Math.min(MAP_SIZE - 2, x2); x++) map[y][x] = 0;
    }
  };

  carve(1, 1, MAP_SIZE - 2, MAP_SIZE - 2);
  for (let y = 2; y < MAP_SIZE - 2; y += 3) {
    for (let x = 2; x < MAP_SIZE - 2; x += 3) {
      if ((x + y + chapterIndex) % 2 === 0) map[y][x] = (chapterIndex % 3) + 2;
    }
  }

  const laneShift = chapterIndex + 1;
  for (let x = 2; x < MAP_SIZE - 2; x++) {
    map[(8 + laneShift) % (MAP_SIZE - 2) + 1][x] = x % 4 === 0 ? 0 : 1;
    map[(14 + laneShift) % (MAP_SIZE - 2) + 1][x] = x % 5 === 0 ? 0 : 1;
  }
  for (let y = 2; y < MAP_SIZE - 2; y++) {
    map[y][(6 + laneShift) % (MAP_SIZE - 2) + 1] = y % 4 === 0 ? 0 : 1;
    map[y][(17 + laneShift) % (MAP_SIZE - 2) + 1] = y % 5 === 0 ? 0 : 1;
  }

  carve(2, 2, 4, 4);
  carve(MAP_SIZE - 5, MAP_SIZE - 5, MAP_SIZE - 3, MAP_SIZE - 3);
  return map;
}

function placePlayer() {
  state.player.x = TILE * 2.5;
  state.player.y = TILE * 2.5;
  state.player.angle = 0;
}

function spawnWave() {
  const ch = chapters[state.chapter];
  const count = Math.floor(6 + state.wave * 3 + state.chapter * 2);
  state.enemies = [];

  for (let i = 0; i < count; i++) {
    const roll = Math.random();
    let type = "trooper";
    if (roll > 0.6) type = "drone";
    if (roll > 0.85) type = "enforcer";
    state.enemies.push(createEnemy(type, ch.enemyScale));
  }

  if (ch.boss && state.wave === ch.waves) state.enemies.push(createEnemy("algorithm", ch.enemyScale));
  if (Math.random() < 0.45) spawnLoreShard();
}

function spawnLoreShard() {
  const pos = findOpenPositionFarFromPlayer();
  state.pickups.push({ x: pos.x, y: pos.y, type: "lore", value: 1, color: "#f2ff73" });
}

function createEnemy(type, scale) {
  const p = enemyProfiles[type];
  const pos = findOpenPositionFarFromPlayer();
  return {
    type,
    x: pos.x,
    y: pos.y,
    hp: p.hp * scale,
    maxHp: p.hp * scale,
    speed: p.speed * (0.9 + Math.random() * 0.3),
    damage: p.damage * scale,
    range: p.range,
    shootMs: p.shootMs,
    color: p.color,
    boss: !!p.boss,
    hackedUntil: 0,
    shotAt: 0,
    phase: Math.random() * Math.PI * 2,
  };
}

function findOpenPositionFarFromPlayer() {
  let x = TILE * 3;
  let y = TILE * 3;
  for (let i = 0; i < 120; i++) {
    const tx = 1 + Math.floor(Math.random() * (MAP_SIZE - 2));
    const ty = 1 + Math.floor(Math.random() * (MAP_SIZE - 2));
    if (state.map[ty][tx] !== 0) continue;
    x = (tx + 0.5) * TILE;
    y = (ty + 0.5) * TILE;
    if (distance(x, y, state.player.x, state.player.y) > TILE * 6) break;
  }
  return { x, y };
}

function loop(ts) {
  if (!state.running) return;
  const dt = Math.min(0.033, ((ts - (loop.last || ts)) || 16.7) / 1000);
  loop.last = ts;
  update(dt, ts);
  draw(ts);
  requestAnimationFrame(loop);
}

function update(dt, now) {
  const p = state.player;
  p.bandwidth = clamp(p.bandwidth + dt * 10, 0, 100);
  state.glitch = Math.max(0, state.glitch - dt * 0.8);

  rotateComms();
  updatePlayerMovement(dt);
  if (mouseDown) fireWeapon(now);

  updateEnemies(dt, now);
  updateProjectiles(dt);
  updatePickups();
  updateParticles(dt);

  if (p.hp <= 0) restartGame();

  if (state.enemies.length === 0 && now > state.chapterTransitionUntil) {
    state.wave += 1;
    if (state.wave > chapters[state.chapter].waves) {
      if (state.chapter === chapters.length - 1) {
        state.running = false;
        const unlocked = state.loreUnlocked.size;
        ui.endingText.textContent = unlocked >= 9
          ? "Jess and Echo merge as a new guardian signal. NeoShanghai wakes, not as clean data, but as free noise."
          : "OmniCore fell, but much was forgotten. Recover more archives next run to uncover Jess' complete ending.";
        ui.victoryOverlay.classList.add("visible");
        return;
      }
      showChapter(state.chapter + 1);
      unlockLore(Math.min(loreFragments.length - 1, state.chapter + 1));
      return;
    }
    spawnWave();
  }
}

function updatePlayerMovement(dt) {
  const p = state.player;
  const stealthing = performance.now() < p.stealthUntil;
  let move = 0;
  let strafe = 0;
  if (keys.has("w")) move += 1;
  if (keys.has("s")) move -= 1;
  if (keys.has("a")) strafe -= 1;
  if (keys.has("d")) strafe += 1;
  if (keys.has("arrowleft")) p.angle -= 2.2 * dt;
  if (keys.has("arrowright")) p.angle += 2.2 * dt;
  p.angle = normalizeAngle(p.angle);

  const sprint = keys.has("shift") ? 1.25 : 1;
  const speed = (stealthing ? 2.8 : 3.8) * TILE * dt * sprint;
  const dx = (Math.cos(p.angle) * move - Math.sin(p.angle) * strafe) * speed;
  const dy = (Math.sin(p.angle) * move + Math.cos(p.angle) * strafe) * speed;
  moveWithCollision(p, dx, dy, 18);

  if (keys.has(" ")) state.glitch = Math.min(1.2, state.glitch + dt * 1.3);
}

function moveWithCollision(obj, dx, dy, radius) {
  const nx = obj.x + dx;
  const ny = obj.y + dy;
  if (!isWall(nx + Math.sign(dx) * radius, obj.y) && !isWall(nx, obj.y + radius) && !isWall(nx, obj.y - radius)) obj.x = nx;
  if (!isWall(obj.x, ny + Math.sign(dy) * radius) && !isWall(obj.x + radius, ny) && !isWall(obj.x - radius, ny)) obj.y = ny;
}

function updateEnemies(dt, now) {
  const p = state.player;
  for (const e of state.enemies) {
    const hacked = now < e.hackedUntil;
    const distToPlayer = distance(e.x, e.y, p.x, p.y);
    let targetX = p.x;
    let targetY = p.y;

    if (hacked) {
      const targetEnemy = state.enemies.find((other) => other !== e && now > other.hackedUntil);
      if (targetEnemy) {
        targetX = targetEnemy.x;
        targetY = targetEnemy.y;
      } else {
        targetX = e.x + Math.cos(now * 0.004 + e.phase) * 32;
        targetY = e.y + Math.sin(now * 0.004 + e.phase) * 32;
      }
    } else if (e.type === "drone") {
      targetX += Math.cos(now * 0.003 + e.phase) * 42;
      targetY += Math.sin(now * 0.003 + e.phase) * 42;
    } else if (e.boss) {
      targetX += Math.sin(now * 0.002) * TILE * 2;
      targetY += Math.cos(now * 0.0025) * TILE * 1.5;
    }

    const angle = Math.atan2(targetY - e.y, targetX - e.x);
    const moveSpeed = e.speed * TILE * dt * (hacked ? 0.75 : 1);
    moveWithCollision(e, Math.cos(angle) * moveSpeed, Math.sin(angle) * moveSpeed, 16);

    const hasLOS = hasLineOfSight(e.x, e.y, targetX, targetY);
    if (now > e.shotAt + e.shootMs && hasLOS) {
      if (hacked) {
        const targetEnemy = state.enemies.find((other) => other !== e && now > other.hackedUntil && distance(other.x, other.y, e.x, e.y) < e.range);
        if (targetEnemy) {
          e.shotAt = now;
          spawnProjectile(e.x, e.y, Math.atan2(targetEnemy.y - e.y, targetEnemy.x - e.x), e.damage * 0.6, false, e.color);
        }
      } else if (distToPlayer < e.range && now > p.stealthUntil) {
        e.shotAt = now;
        const shots = e.boss ? 3 : 1;
        for (let i = 0; i < shots; i++) {
          const spread = (Math.random() - 0.5) * (e.boss ? 0.25 : 0.08);
          spawnProjectile(e.x, e.y, Math.atan2(p.y - e.y, p.x - e.x) + spread, e.damage, true, e.color);
        }
      }
    }
  }

  state.enemies = state.enemies.filter((e) => {
    if (e.hp > 0) return true;
    state.player.kills += 1;
    spawnSpark(e.x, e.y, e.color, 22);
    maybeDropPickup(e.x, e.y);
    return false;
  });
}

function spawnProjectile(x, y, angle, damage, hostile, color) {
  state.projectiles.push({ x, y, vx: Math.cos(angle) * TILE * 7, vy: Math.sin(angle) * TILE * 7, damage, hostile, color, life: 1.1, radius: 5 });
}

function updateProjectiles(dt) {
  const p = state.player;
  for (const b of state.projectiles) {
    b.x += b.vx * dt;
    b.y += b.vy * dt;
    b.life -= dt;
    if (isWall(b.x, b.y)) {
      b.life = 0;
      continue;
    }

    if (b.hostile) {
      if (distance(b.x, b.y, p.x, p.y) < 16) {
        p.hp = clamp(p.hp - b.damage * 0.18, 0, p.maxHp);
        state.glitch = Math.min(1.5, state.glitch + 0.18);
        b.life = 0;
        spawnSpark(p.x, p.y, "#ff5b7b", 10);
      }
    } else {
      for (const e of state.enemies) {
        if (distance(b.x, b.y, e.x, e.y) < 18) {
          e.hp -= b.damage;
          b.life = 0;
          spawnSpark(e.x, e.y, b.color, 12);
          break;
        }
      }
    }
  }
  state.projectiles = state.projectiles.filter((b) => b.life > 0);
}

function maybeDropPickup(x, y) {
  const roll = Math.random();
  if (roll < 0.16) state.pickups.push({ x, y, type: "ammo", value: 20, color: "#5be4ff" });
  else if (roll < 0.30) state.pickups.push({ x, y, type: "health", value: 18, color: "#ff668f" });
  else if (roll < 0.44) state.pickups.push({ x, y, type: "bandwidth", value: 22, color: "#7dff8f" });
  else if (roll < 0.52) state.pickups.push({ x, y, type: "lore", value: 1, color: "#f2ff73" });
}

function updatePickups() {
  const p = state.player;
  state.pickups = state.pickups.filter((pick) => {
    if (distance(pick.x, pick.y, p.x, p.y) > 22) return true;

    if (pick.type === "health") p.hp = clamp(p.hp + pick.value, 0, p.maxHp);
    if (pick.type === "bandwidth") p.bandwidth = clamp(p.bandwidth + pick.value, 0, 100);
    if (pick.type === "ammo") {
      for (let i = 0; i < weapons.length; i++) {
        if (!Number.isFinite(weapons[i].reserve)) continue;
        p.reserve[i] += pick.value;
      }
    }
    if (pick.type === "lore") {
      unlockLore(nextLockedLoreIndex());
      rotateComms(true);
    }
    return false;
  });
}

function nextLockedLoreIndex() {
  for (let i = 0; i < loreFragments.length; i++) {
    if (!state.loreUnlocked.has(i)) return i;
  }
  return loreFragments.length - 1;
}

function unlockLore(index) {
  if (index < 0 || index >= loreFragments.length || state.loreUnlocked.has(index)) return;
  state.loreUnlocked.add(index);
  const title = `Archive ${String(index + 1).padStart(2, "0")}`;
  state.loreFeed.unshift(title);
  state.loreFeed = state.loreFeed.slice(0, 5);
  renderLoreFeed();
  ui.commsSpeaker.textContent = "ARCHIVE";
  ui.commsText.textContent = loreFragments[index];
  state.commsUntil = performance.now() + 9000;
}

function renderLoreFeed() {
  ui.loreCounter.textContent = `Fragments Unlocked: ${state.loreUnlocked.size} / ${loreFragments.length}`;
  ui.loreFeed.innerHTML = "";
  for (const title of state.loreFeed) {
    const li = document.createElement("li");
    li.textContent = title;
    ui.loreFeed.appendChild(li);
  }
}

function updateParticles(dt) {
  for (const p of state.particles) {
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.life -= dt;
  }
  state.particles = state.particles.filter((p) => p.life > 0);
}

function fireWeapon(now) {
  const p = state.player;
  const w = weapons[p.weaponIndex];
  if (now < p.reloadingUntil || now - p.lastShotAt < w.fireRate) return;

  if (w.melee) {
    p.lastShotAt = now;
    let didHit = false;
    for (const e of state.enemies) {
      const d = distance(p.x, p.y, e.x, e.y);
      const dir = Math.atan2(e.y - p.y, e.x - p.x);
      const diff = Math.abs(angleDiff(p.angle, dir));
      if (d < w.range && diff < 0.5) {
        e.hp -= w.damage;
        didHit = true;
        spawnSpark(e.x, e.y, w.color, 16);
      }
    }
    if (!didHit) spawnSpark(p.x + Math.cos(p.angle) * 26, p.y + Math.sin(p.angle) * 26, w.color, 8);
    state.muzzleFlashUntil = now + 100;
    return;
  }

  if (p.ammoInMag[p.weaponIndex] <= 0) return;
  p.lastShotAt = now;
  p.ammoInMag[p.weaponIndex] -= 1;
  state.muzzleFlashUntil = now + 80;

  for (let i = 0; i < w.pellets; i++) {
    const spread = (Math.random() - 0.5) * w.spread;
    castHitscan(p.angle + spread, w.damage, w.range, w.color);
  }
}

function castHitscan(angle, damage, range, color) {
  const p = state.player;
  let hitX = p.x;
  let hitY = p.y;
  for (let t = 0; t < range; t += 4) {
    const x = p.x + Math.cos(angle) * t;
    const y = p.y + Math.sin(angle) * t;
    hitX = x;
    hitY = y;
    if (isWall(x, y)) {
      spawnSpark(hitX, hitY, color, 6);
      return;
    }
    for (const e of state.enemies) {
      if (distance(x, y, e.x, e.y) < (e.boss ? 24 : 16)) {
        e.hp -= damage;
        spawnSpark(e.x, e.y, color, 12);
        return;
      }
    }
  }
  spawnSpark(hitX, hitY, color, 4);
}

function reloadWeapon() {
  const p = state.player;
  const w = weapons[p.weaponIndex];
  if (!Number.isFinite(w.mag)) return;
  if (performance.now() < p.reloadingUntil) return;
  if (p.ammoInMag[p.weaponIndex] >= w.mag || p.reserve[p.weaponIndex] <= 0) return;

  p.reloadingUntil = performance.now() + w.reloadMs;
  setTimeout(() => {
    const need = w.mag - p.ammoInMag[p.weaponIndex];
    const amount = Math.min(need, p.reserve[p.weaponIndex]);
    p.ammoInMag[p.weaponIndex] += amount;
    p.reserve[p.weaponIndex] -= amount;
  }, w.reloadMs);
}

function castHack() {
  const p = state.player;
  const now = performance.now();
  if (now < p.hackCooldownUntil || p.bandwidth < 28) return;
  p.bandwidth -= 28;
  p.hackCooldownUntil = now + 9000;
  for (const e of state.enemies) {
    if (distance(p.x, p.y, e.x, e.y) < TILE * 4.6 && (e.type === "drone" || e.type === "enforcer")) {
      e.hackedUntil = now + 5000;
      spawnSpark(e.x, e.y, "#84ffb6", 18);
    }
  }
  ui.commsSpeaker.textContent = "ECHO";
  ui.commsText.textContent = "Hijack successful. Their loyalty stack is temporarily mine.";
  state.commsUntil = performance.now() + 5000;
}

function castStealth() {
  const p = state.player;
  const now = performance.now();
  if (now < p.stealthCooldownUntil || p.bandwidth < 25) return;
  p.bandwidth -= 25;
  p.stealthUntil = now + 4200;
  p.stealthCooldownUntil = now + 12000;
}

function draw(now) {
  const realmDigital = keys.has(" ");
  drawSkyAndFloor(realmDigital, now);
  drawWorld(now);
  drawProjectilesAndPickups();
  drawWeaponView(now);
  drawRain();
  drawMinimap();
  drawCrosshair();
  drawParticlesOverlay();
  updateUI(now);
}

function drawSkyAndFloor(realmDigital, now) {
  const sky = ctx.createLinearGradient(0, 0, 0, HALF_H);
  if (realmDigital) {
    sky.addColorStop(0, "#081d2c");
    sky.addColorStop(1, "#144441");
  } else {
    sky.addColorStop(0, "#0e1335");
    sky.addColorStop(1, "#281645");
  }
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, WIDTH, HALF_H);

  const floor = ctx.createLinearGradient(0, HALF_H, 0, HEIGHT);
  floor.addColorStop(0, realmDigital ? "#061518" : "#0d0f1a");
  floor.addColorStop(1, "#030408");
  ctx.fillStyle = floor;
  ctx.fillRect(0, HALF_H, WIDTH, HALF_H);

  const pulse = 0.05 + Math.sin(now * 0.0015) * 0.02;
  ctx.fillStyle = `rgba(70,230,255,${pulse})`;
  for (let y = HALF_H + 10; y < HEIGHT; y += 18) ctx.fillRect(0, y, WIDTH, 1);
}

function drawWorld(now) {
  const p = state.player;
  let rayAngle = p.angle - FOV / 2;
  let bufferIndex = 0;

  for (let x = 0; x < WIDTH; x += RAY_STEP) {
    const hit = castRay(rayAngle);
    const corrected = hit.distance * Math.cos(rayAngle - p.angle);
    state.depthBuffer[bufferIndex++] = corrected;
    const wallHeight = Math.min(HEIGHT, (TILE * 700) / (corrected + 0.0001));
    const top = HALF_H - wallHeight / 2;
    const shade = clamp(1 - corrected / (MAX_DEPTH * 1.1), 0.08, 1);
    const hue = 185 + hit.tile * 12;
    ctx.fillStyle = `hsla(${hue}, 85%, ${22 + shade * 35}%, 1)`;
    ctx.fillRect(x, top, RAY_STEP + 1, wallHeight);
    if (hit.vertical) {
      ctx.fillStyle = `rgba(0,0,0,${0.15 + (1 - shade) * 0.35})`;
      ctx.fillRect(x, top, RAY_STEP + 1, wallHeight);
    }
    rayAngle += (FOV / WIDTH) * RAY_STEP;
  }

  drawEnemySprites(now);
}

function castRay(angle) {
  const p = state.player;
  let d = 0;
  let x = p.x;
  let y = p.y;
  let lastX = x;
  let lastY = y;
  while (d < MAX_DEPTH) {
    x += Math.cos(angle) * 4;
    y += Math.sin(angle) * 4;
    d += 4;
    if (isWall(x, y)) {
      const tile = tileAt(x, y);
      const vertical = Math.abs(Math.floor(x / TILE) - Math.floor(lastX / TILE)) > Math.abs(Math.floor(y / TILE) - Math.floor(lastY / TILE));
      return { distance: d, tile, vertical };
    }
    lastX = x;
    lastY = y;
  }
  return { distance: MAX_DEPTH, tile: 1, vertical: false };
}

function drawEnemySprites(now) {
  const p = state.player;
  const sprites = [];
  for (const e of state.enemies) {
    const dx = e.x - p.x;
    const dy = e.y - p.y;
    const dist = Math.hypot(dx, dy);
    const rel = angleDiff(p.angle, Math.atan2(dy, dx));
    if (Math.abs(rel) > FOV * 0.8 || dist < 16) continue;

    const size = clamp((TILE * 540) / dist, 14, HEIGHT * 0.9) * (e.boss ? 1.6 : 1);
    const screenX = (rel / FOV + 0.5) * WIDTH;
    const y = HALF_H - size * 0.5;
    sprites.push({ e, dist, screenX, y, size });
  }

  sprites.sort((a, b) => b.dist - a.dist);
  for (const s of sprites) {
    const depthSample = clamp(Math.floor(s.screenX / RAY_STEP), 0, state.depthBuffer.length - 1);
    if (state.depthBuffer[depthSample] < s.dist) continue;

    const hacked = now < s.e.hackedUntil;
    const alpha = now < state.player.stealthUntil && !hacked ? 0.8 : 1;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = hacked ? "#95ffb8" : s.e.color;
    ctx.fillRect(s.screenX - s.size * 0.35, s.y, s.size * 0.7, s.size);

    const eyePulse = 0.3 + Math.sin(now * 0.01 + s.e.phase) * 0.2;
    ctx.fillStyle = `rgba(255,255,255,${0.7 + eyePulse})`;
    ctx.fillRect(s.screenX - s.size * 0.12, s.y + s.size * 0.26, s.size * 0.24, s.size * 0.1);
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(s.screenX - s.size * 0.3, s.y - 10, s.size * 0.6, 6);
    ctx.fillStyle = "#6bff9b";
    ctx.fillRect(s.screenX - s.size * 0.3, s.y - 10, (s.size * 0.6) * (s.e.hp / s.e.maxHp), 6);
    ctx.globalAlpha = 1;
  }
}

function drawProjectilesAndPickups() {
  const p = state.player;
  for (const b of state.projectiles) {
    const dx = b.x - p.x;
    const dy = b.y - p.y;
    const dist = Math.hypot(dx, dy);
    const rel = angleDiff(p.angle, Math.atan2(dy, dx));
    if (Math.abs(rel) > FOV * 0.8 || dist < 8) continue;

    const sx = (rel / FOV + 0.5) * WIDTH;
    const size = clamp((TILE * 140) / dist, 2, 20);
    const y = HALF_H - size * 0.5;
    const depthIndex = clamp(Math.floor(sx / RAY_STEP), 0, state.depthBuffer.length - 1);
    if (state.depthBuffer[depthIndex] > dist) {
      ctx.fillStyle = b.color;
      ctx.fillRect(sx - size / 2, y, size, size);
    }
  }

  for (const pick of state.pickups) {
    const dx = pick.x - p.x;
    const dy = pick.y - p.y;
    const dist = Math.hypot(dx, dy);
    const rel = angleDiff(p.angle, Math.atan2(dy, dx));
    if (Math.abs(rel) > FOV * 0.8 || dist < 8) continue;
    const sx = (rel / FOV + 0.5) * WIDTH;
    const size = clamp((TILE * 130) / dist, 4, 22);
    const y = HALF_H + 28 - size;
    ctx.fillStyle = pick.color;
    ctx.fillRect(sx - size / 2, y, size, size);
  }
}

function drawWeaponView(now) {
  const p = state.player;
  const w = weapons[p.weaponIndex];
  const bob = Math.sin(now * 0.012) * 5;
  const x = WIDTH * 0.58;
  const y = HEIGHT - 160 + bob;

  ctx.fillStyle = "rgba(8,12,24,0.75)";
  ctx.fillRect(x - 10, y - 50, 310, 160);
  ctx.fillStyle = w.color;
  ctx.fillRect(x + 10, y + 32, 220, 26);
  ctx.fillStyle = "#1d243b";
  ctx.fillRect(x + 180, y + 20, 95, 48);
  ctx.fillStyle = "#68c5ff";
  ctx.fillRect(x + 190, y + 30, 74, 10);

  if (now < state.muzzleFlashUntil) {
    ctx.fillStyle = `rgba(255,255,255,${0.45 + Math.random() * 0.2})`;
    ctx.beginPath();
    ctx.arc(x + 270, y + 45, 35, 0, Math.PI * 2);
    ctx.fill();
  }

  if (performance.now() < p.reloadingUntil) {
    const left = p.reloadingUntil - performance.now();
    ctx.fillStyle = "#ffe37d";
    ctx.fillText(`RELOADING ${Math.ceil(left / 100) / 10}s`, x + 20, y - 14);
  }
}

function drawRain() {
  ctx.strokeStyle = "rgba(130,220,255,0.25)";
  for (const d of state.rain) {
    d.y += d.z * 5;
    d.x -= d.z * 0.8;
    if (d.y > HEIGHT) {
      d.y = -10;
      d.x = Math.random() * WIDTH;
    }
    if (d.x < 0) d.x = WIDTH;
    ctx.beginPath();
    ctx.moveTo(d.x, d.y);
    ctx.lineTo(d.x + d.z, d.y + d.z * 9);
    ctx.stroke();
  }
}

function drawCrosshair() {
  const cx = WIDTH / 2;
  const cy = HEIGHT / 2;
  ctx.strokeStyle = "rgba(120,250,255,0.95)";
  ctx.beginPath();
  ctx.moveTo(cx - 12, cy); ctx.lineTo(cx - 4, cy);
  ctx.moveTo(cx + 4, cy); ctx.lineTo(cx + 12, cy);
  ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy - 4);
  ctx.moveTo(cx, cy + 4); ctx.lineTo(cx, cy + 12);
  ctx.stroke();
}

function drawMinimap() {
  const size = 170;
  const scale = size / (MAP_SIZE * TILE);
  const x0 = 18;
  const y0 = HEIGHT - size - 18;

  ctx.fillStyle = "rgba(2,8,20,0.72)";
  ctx.fillRect(x0, y0, size, size);

  for (let y = 0; y < MAP_SIZE; y++) {
    for (let x = 0; x < MAP_SIZE; x++) {
      if (state.map[y][x] === 0) continue;
      ctx.fillStyle = "rgba(122, 120, 175, 0.65)";
      ctx.fillRect(x0 + x * TILE * scale, y0 + y * TILE * scale, TILE * scale, TILE * scale);
    }
  }

  for (const e of state.enemies.slice(0, 60)) {
    ctx.fillStyle = performance.now() < e.hackedUntil ? "#84ffbb" : "#ff728f";
    ctx.fillRect(x0 + e.x * scale - 1, y0 + e.y * scale - 1, 3, 3);
  }
  for (const pick of state.pickups) {
    ctx.fillStyle = pick.color;
    ctx.fillRect(x0 + pick.x * scale - 1, y0 + pick.y * scale - 1, 2, 2);
  }

  const p = state.player;
  ctx.fillStyle = "#4fffff";
  ctx.beginPath();
  ctx.arc(x0 + p.x * scale, y0 + p.y * scale, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#4fffff";
  ctx.beginPath();
  ctx.moveTo(x0 + p.x * scale, y0 + p.y * scale);
  ctx.lineTo(x0 + p.x * scale + Math.cos(p.angle) * 10, y0 + p.y * scale + Math.sin(p.angle) * 10);
  ctx.stroke();
}

function drawParticlesOverlay() {
  for (const p of state.particles) {
    ctx.globalAlpha = clamp(p.life / 0.6, 0, 1);
    ctx.fillStyle = p.color;
    ctx.fillRect(p.x, p.y, 2.5, 2.5);
  }
  ctx.globalAlpha = 1;

  if (state.glitch > 0) {
    ctx.fillStyle = `rgba(255,40,90,${state.glitch * 0.09})`;
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    for (let i = 0; i < 6; i++) {
      const y = Math.random() * HEIGHT;
      ctx.fillStyle = `rgba(80,255,255,${state.glitch * 0.07})`;
      ctx.fillRect(Math.random() * 24, y, WIDTH - Math.random() * 40, 2);
    }
  }
}

function updateUI(now) {
  const p = state.player;
  const w = weapons[p.weaponIndex];
  ui.healthFill.style.width = `${clamp((p.hp / p.maxHp) * 100, 0, 100)}%`;
  ui.healthText.textContent = `${Math.ceil(p.hp)}`;
  ui.bandwidthFill.style.width = `${p.bandwidth}%`;
  ui.bandwidthText.textContent = `${Math.floor(p.bandwidth)}`;
  ui.weaponName.textContent = w.name;
  const ammo = Number.isFinite(w.mag) ? `${p.ammoInMag[p.weaponIndex]} / ${p.reserve[p.weaponIndex]}` : "∞";
  ui.ammo.textContent = `Ammo: ${ammo}`;

  const qcd = Math.max(0, (p.hackCooldownUntil - now) / 1000);
  const ecd = Math.max(0, (p.stealthCooldownUntil - now) / 1000);
  ui.cooldowns.textContent = `Hack [Q]: ${qcd <= 0 ? "READY" : qcd.toFixed(1) + "s"} | Stealth [E]: ${ecd <= 0 ? "READY" : ecd.toFixed(1) + "s"}`;
  ui.wave.textContent = `Chapter ${state.chapter + 1} · Wave ${state.wave} · Hostiles ${state.enemies.length}`;
  renderLoreFeed();
}

function spawnSpark(x, y, color, count) {
  for (let i = 0; i < count; i++) {
    state.particles.push({ x, y, vx: (Math.random() - 0.5) * 80, vy: (Math.random() - 0.5) * 80, life: 0.25 + Math.random() * 0.4, color });
  }
}

function hasLineOfSight(x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const dist = Math.hypot(dx, dy);
  const steps = Math.ceil(dist / 8);
  for (let i = 1; i < steps; i++) {
    const t = i / steps;
    if (isWall(x1 + dx * t, y1 + dy * t)) return false;
  }
  return true;
}

function tileAt(x, y) {
  const tx = Math.floor(x / TILE);
  const ty = Math.floor(y / TILE);
  if (tx < 0 || ty < 0 || tx >= MAP_SIZE || ty >= MAP_SIZE) return 1;
  return state.map[ty][tx];
}

function isWall(x, y) {
  return tileAt(x, y) !== 0;
}

function normalizeAngle(a) {
  while (a <= -Math.PI) a += Math.PI * 2;
  while (a > Math.PI) a -= Math.PI * 2;
  return a;
}

function angleDiff(a, b) {
  return normalizeAngle(b - a);
}

function distance(x1, y1, x2, y2) {
  return Math.hypot(x2 - x1, y2 - y1);
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}
