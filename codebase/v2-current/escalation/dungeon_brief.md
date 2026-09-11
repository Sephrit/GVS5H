Build this game.

Dungeon Crawl -- a first-person 3D dungeon explorer that runs in a web browser.

- First-person view. WASD moves the player; the mouse looks around after the player clicks
  to capture the pointer.
- A new dungeon is generated every time the page loads: 8 to 12 rectangular stone rooms
  joined by corridors.
- The dungeon is dark except for a flickering torch the player carries.
- The player cannot walk through walls.
- A key is hidden in one room and a locked exit door is in another. Picking up the key and
  then reaching the door wins the game; show a clear win message.
- A small minimap in a corner of the screen fills in as the player explores.

Technical requirements:

- One self-contained HTML file. Load three.js as an ES module through an import map:
  "three" -> https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js and
  "three/addons/" -> https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/
  Use no other external files, images or sounds; generate any textures in code.
- For automated testing, define window.game and keep it current every frame:
  window.game.player = {x, y, z} (the player's world position), window.game.rooms = the
  number of rooms generated, window.game.hasKey and window.game.won = booleans.
- Movement must work by holding keys (track keydown/keyup and move every frame). Automated
  tests cannot lock the pointer, so WASD must also move the player while the pointer is not
  locked.
