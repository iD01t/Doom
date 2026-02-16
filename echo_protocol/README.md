# SYNTAX_ERROR: Echo Protocol (Doom-style Browser Build)

A Doom-inspired raycast FPS with expanded lore, persona presentation, and cyberpunk narrative framing.

## Implemented systems

- Classic-style **raycast renderer** (pseudo-3D walls, depth buffer, sprite projection)
- **5 chapters** with escalating multi-wave progression and final boss phase
- **4 weapons** with distinct behavior, damage profiles, spread, magazines, reserve ammo, and reload timing
- **Abilities**
  - `Q` Hack pulse (temporarily converts drones/enforcers)
  - `E` Stealth cloak (temporary detection drop)
- Enemy ecosystem
  - Trooper, Drone, Enforcer, and Golden Algorithm boss
  - AI pursuit + line-of-sight shooting + hacked ally behavior
- Combat feedback
  - Hitscan firing, enemy projectiles, particles, muzzle flashes, glitch damage effects
- Progression support
  - Pickup drops (health, bandwidth, ammo)
  - Weapon unlocks by chapter
- **Narrative/Lore systems**
  - Root Access comms panel with chapter-specific dialogue (Jess / Kade / Echo / Algorithm)
  - Archive fragment pickup system unlocking Damaged Archives lore entries
  - Persona cards in intro overlay and chapter quote stingers
  - Ending text variant based on lore fragment completion
- Full HUD
  - Health/bandwidth bars, ammo, cooldowns, chapter-wave-hostiles readout, minimap
  - Comms + archive feed panels for in-run story context
- Atmosphere
  - Neon cyberpunk palette, rain overlay, digital realm distortion mode (`Space`)

## Controls

- **Move:** `W A S D`
- **Turn:** Mouse (click canvas to lock pointer) or `←` `→`
- **Fire:** Left Mouse
- **Reload:** `R`
- **Hack:** `Q`
- **Stealth:** `E`
- **Realm Shift / Digital pulse:** Hold `Space`
- **Weapon select:** `1` `2` `3` `4`

## Run

```bash
cd echo_protocol
python3 -m http.server 8000
```

Open: <http://localhost:8000>

> Optional audio: add `track.mp3` in this folder for looping background music.
