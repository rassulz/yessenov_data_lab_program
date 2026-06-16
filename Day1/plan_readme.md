# Dungeon Logic

This document stores the full agreed direction for the game, including the original concept and the later visual and menu refinements.

## Project Identity

The official game name is `Dungeon Logic`.

This project is a 2D browser game built as a website. The player controls a character inside a dangerous mine or dungeon chamber. A math problem appears on the wall behind the character. In front of the character there are three downward wells or tunnel-shafts. Each well has an answer sign, but only one answer is correct.

The player must choose the correct well.

- If the player chooses the correct answer, the hero falls safely into the selected shaft and reaches the next level.
- If the player chooses the wrong answer, the hero still falls, but dies from the impact and the game restarts from Level 1.

The game should feel atmospheric, dramatic, memorable, and easy to understand immediately.

## Core Gameplay

The game is based on one simple but tense loop:

1. Enter a chamber.
2. Read the math problem on the cave wall.
3. Compare the three possible answers.
4. Choose one well.
5. Survive and descend deeper, or fail and restart.

The tension should come from presentation and consequences, not from complicated controls.

## Level Structure

The first release must contain 10 levels.

Requested difficulty rules:

- Levels 1-4: addition with small one-digit numbers.
- From Level 5: subtraction starts to appear.
- From Level 9: binary numbers begin.

### Level progression table

| Level | Math focus | Example |
| --- | --- | --- |
| 1 | Easy addition | `2 + 3 = ?` |
| 2 | Easy addition | `4 + 1 = ?` |
| 3 | Easy addition | `6 + 2 = ?` |
| 4 | Slightly larger addition | `7 + 5 = ?` |
| 5 | First subtraction | `9 - 4 = ?` |
| 6 | Larger subtraction | `13 - 5 = ?` |
| 7 | Mixed addition | `8 + 9 = ?` |
| 8 | Mixed subtraction | `17 - 8 = ?` |
| 9 | Binary addition | `101 + 10 = ?` |
| 10 | Binary final | `111 - 1 = ?` |

### Answer rules

Each level should contain:

- 1 correct answer
- 2 believable wrong answers

Wrong answers should be close enough to create tension.

Good patterns:

- `correct + 1`
- `correct - 1`
- common human mistake result
- binary-looking fake answers for binary levels

## Gameplay Scene Layout

Each level chamber should include:

- A player character standing near the center
- A stone wall behind the character
- A visible math problem written on the wall
- Three wells below or in front of the character
- A sign above or in front of each well showing one answer
- Torches, rubble, cracks, dust, chains, or bones for dungeon atmosphere
- More obvious cave and mine dressing such as rocks, stalactites, lava cracks, and stone formations

### Composition rules

- The scene should be easy to read in a single glance.
- The equation should be immediately visible.
- The three answers should be clearly separated.
- The wells should feel like dangerous vertical shafts, closer to wells than flat doors.

## Confirmed Art Direction

The style must be:

- Pixel art
- Inspired by arcade games from the 1990s
- Fantasy dungeon or cave adventure
- Atmospheric but still readable
- Strongly stylized, not modern-flat and not generic casual-game UI

### Important visual rule

The project should feel like a retro arcade fantasy game, not like a clean modern website with game text on top.

### Visual goals

- Sharp pixel feeling
- Bold silhouettes
- Strong contrast
- Warm torch and lava lighting
- Dark cave stone mixed with bright hot accents
- A dramatic dungeon tone

## Confirmed Color and Background Direction

The cave background must be more colorful and alive, not plain dark gray.

The approved background direction is:

- A cave or dungeon environment
- Lava at the bottom or in the distance
- Warm mysterious light
- Firelit orange and gold glow
- Rich stone colors, not a flat black background
- Extra fantasy cave details like crystals, lava falls, sparks, smoke, or glowing reflections
- Clear rock shapes and cave massing, not only abstract dark gradients
- Stalactites hanging from the top of the cave
- More visible stone clusters and underground depth
- Lava cracks or hot glowing breaks in the ground when useful

### Background mood

The background should feel like a magical dangerous cavern:

- warm
- dramatic
- mysterious
- colorful
- retro arcade

### Confirmed cave details

The user explicitly requested that the cave and dungeon feeling should be stronger.

Important background elements to preserve:

- rocks
- lava
- stalactites
- cave walls
- deeper underground mine feeling

## Main Menu Plan

The menu is the first priority and should feel strong, centered, and atmospheric.

### Confirmed menu requirements

- Everything important should be centered on the screen.
- The game title must be very visible.
- The game title must say `Dungeon Logic`.
- There must be a clear and visible `Play Game` or `Start Game` button.
- The play button should sit directly under the main menu content.
- Under the menu, there should be a simple instruction block explaining how to play.
- The menu should not rely on hiding the instructions in a modal as the main approach.

### Confirmed menu structure

Top to bottom:

1. Small retro eyebrow text
2. Large visible `Dungeon Logic` title
3. Short atmospheric tagline
4. Big centered `Play Game` button
5. Secondary sound and music buttons
6. Simple centered instruction block directly below the menu controls
7. Small status text or atmospheric message

### Menu visual style

- Pixel-art fantasy arcade look
- Stone or dungeon UI panels
- Strong retro button shapes
- Warm highlights from fire and lava
- Torches on the sides
- Centered composition

### Menu background style

- Cave walls
- Lava
- Warm mysterious lighting
- Pixelated or blocky cave forms
- More color depth than a plain dark background
- Visible stone and cave texture
- Stronger dungeon and mine identity

## Confirmed Menu UX Details

The menu must communicate the game instantly.

The player should understand in a few seconds:

- what the game is called
- where to click to start
- what the basic rule of the game is

### Instruction block under menu

The inline instruction section should explain:

1. Read the math problem on the wall.
2. Look at the three wells and their answers.
3. Choose the correct well.
4. A correct answer leads to the next level.
5. A wrong answer causes death and restarts the game.

It should sit directly below the menu rather than feeling hidden.

## Audio Direction

The game should include:

- Dark fantasy ambience
- Torch crackle
- Cave wind
- Lava or fire ambience
- Success sting
- Failure impact
- UI button click

Menu audio should feel ominous and adventurous, not cheerful.

## Character Direction

The player character should now be treated primarily as a pixel-art miner, not just a generic adventurer.

### Confirmed miner direction

The user explicitly asked for the character to look more like a miner.

Important traits:

- miner helmet
- lamp on the helmet
- work clothes or mining jacket
- boots
- rugged underground worker silhouette
- readable at small size

Optional supporting details:

- beard
- backpack
- pickaxe, hammer, or mining tool
- belt or work gear

The final silhouette should immediately read as `miner in a dungeon mine`.

## Website Implementation Direction

The game must be implemented as a website.

Recommended long-term stack:

- `Phaser 3`
- `TypeScript`
- `Vite`

Current prototype direction:

- Build the menu first
- Keep layout centered
- Preserve the pixel-art arcade look in HTML/CSS or in the future Phaser UI scene

## Scene Architecture

Suggested scenes:

- `BootScene`
- `MenuScene`
- `GameScene`
- `TransitionScene`
- `GameOverScene`
- `VictoryScene`

### MenuScene requirements

`MenuScene` should include:

- centered title
- big visible play button
- sound and music toggles
- inline instruction block under the menu
- cave and lava background
- stronger mine and dungeon atmosphere with visible rock shapes and cave depth

### GameScene requirements

`GameScene` should include:

- a visible cave chamber
- stronger mine and dungeon styling
- rocks and stone clusters
- lava and hot glow
- stalactites from the cave ceiling
- a miner character instead of a generic fantasy hero

## Responsive Rules

The game should work on desktop and mobile browsers.

Important responsive requirements:

- The menu should remain centered.
- The title should remain readable.
- The `Play Game` button should remain obvious.
- The instruction block should remain under the menu.
- The three answer wells in gameplay must remain easy to tap.

## Inspirations

The game should take mood and presentation inspiration from retro dungeon and arcade-style fantasy games, while keeping the core math-and-wells mechanic original.

Useful inspiration references:

1. `Spelunky Classic`
2. `Pixel Dungeon`
3. `Shovel Knight`
4. `Darkest Dungeon`

These are references for atmosphere, readability, and pacing only. No direct asset copying.

## Current Approved Summary

The final agreed direction so far is:

- The game name is `Dungeon Logic`.
- It is a browser game built as a website.
- The visual style should feel like a pixel-art 1990s arcade fantasy game.
- The cave background must be colorful and atmospheric.
- Lava and warm mysterious light are important.
- The cave should visibly include rocks, dungeon stone, and stalactites.
- The environment should feel more like a mine and underground cavern, not just a dark abstract background.
- The menu must be centered.
- The title must be large and clearly visible.
- The menu must contain a clear `Play Game` button.
- The instructions must appear under the menu, not mainly inside a popup.
- The main character should look like a miner, with gear that reads clearly as mining equipment.
- The gameplay concept remains the same: choose the correct well based on the wall math problem and survive all 10 levels.
