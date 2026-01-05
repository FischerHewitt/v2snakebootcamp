"use client";

import { useEffect, useRef } from "react";
import { io, Socket } from "socket.io-client";

const HEADER_HEIGHT_PX = 64;

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<Socket | undefined>(undefined);

  type Point = { x: number; y: number };

  const gridRef  = useRef({ w: 20, h: 20, cell: 24 }); // board size (cells) and cell pixel size
  const snakeRef = useRef<Point[]>([]);                 // snake body segments
  const foodRef  = useRef<Point>({ x: 5, y: 5 });       // food location
  const aliveRef = useRef(true);                        // alive / game over
  const scoreRef = useRef(0);                           // current score

  // Describe what we expect from the server frames
  // (kept near top so it's visible to both effects)
  type ServerState = {
    grid_width: number;
    grid_height: number;
    game_tick: number;
    snake: [number, number][];  // array of [x, y]
    food: [number, number];     // [x, y]
    score: number;
    alive?: boolean;
  };

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { w, h, cell } = gridRef.current;

    // keep the canvas pixel size in sync with the grid
    const width = w * cell;
    const height = h * cell;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;

    ctx.clearRect(0, 0, width, height);

    // background
    ctx.fillStyle = "#0b0b0b";
    ctx.fillRect(0, 0, width, height);

    // faint grid (visual guide)
    ctx.globalAlpha = 0.15;
    for (let gx = 0; gx <= w; gx++) ctx.fillRect(gx * cell, 0, 1, height);
    for (let gy = 0; gy <= h; gy++) ctx.fillRect(0, gy * cell, width, 1);
    ctx.globalAlpha = 1;

    // snake
    ctx.fillStyle = "#00ffeeff";
    for (const seg of snakeRef.current) {
      ctx.fillRect(seg.x * cell + 1, seg.y * cell + 1, cell - 2, cell - 2);
    }

    // food
    ctx.fillStyle = "#ef4444";
    ctx.fillRect(foodRef.current.x * cell + 1, foodRef.current.y * cell + 1, cell - 2, cell - 2);

    // overlay text: score + game status
    ctx.fillStyle = "white";
    ctx.font = Math.max(14, Math.floor(cell * 0.6)) + "px sans-serif";
    ctx.fillText(`Score: ${scoreRef.current}${aliveRef.current ? "" : " — GAME OVER"}`, 10, 20);
  }

  // SOCKET: open connection, start game, receive frames, and clean up
  useEffect(() => {
    if (socketRef.current !== undefined) return; // already connected

    // open the socket using pure websocket transport
    const s = io("http://localhost:8765", { transports: ["websocket"] });
    socketRef.current = s;

    const onConnect = () => {
      s.emit("start_game", {
        grid_width: gridRef.current.w,   // number of columns
        grid_height: gridRef.current.h,  // number of rows
        tick_ms: 120,                    // 1 step every 120ms
      });
    };

    const onGameState = (data: unknown) => {
      const st = data as Partial<ServerState>;

      if (typeof st.grid_width === "number") gridRef.current.w = st.grid_width;
      if (typeof st.grid_height === "number") gridRef.current.h = st.grid_height;
      if (typeof st.score === "number") scoreRef.current = st.score;
      if (typeof st.alive === "boolean") aliveRef.current = st.alive;

      if (Array.isArray(st.snake)) {
        snakeRef.current = st.snake.map(([x, y]) => ({ x, y }));
      }
      if (Array.isArray(st.food)) {
        const [x, y] = st.food;
        foodRef.current = { x, y };
      }

      draw(); // repaint with latest data
    };

    // separate event for final frame
    const onGameOver = (data: unknown) => {
      const st = data as Partial<ServerState>;
      if (typeof st.score === "number") scoreRef.current = st.score;
      aliveRef.current = false;
      draw();
    };

    s.on("connect", onConnect);

    // Enhancement: be robust to either event name
    s.on("game_state", onGameState);
    s.on("update", onGameState); // if backend used "update" instead

    s.on("game_over", onGameOver);

    return () => {
      s.off("connect", onConnect);
      s.off("game_state", onGameState);
      s.off("update", onGameState);
      s.off("game_over", onGameOver);
      s.disconnect();
      socketRef.current = undefined;
    };
  }, []);

  // Keyboard controls (Arrow keys + WASD → send "change_direction" to server)
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const map: Record<string, string> = {
        ArrowUp: "UP",
        ArrowDown: "DOWN",
        ArrowLeft: "LEFT",
        ArrowRight: "RIGHT",
        w: "UP",
        s: "DOWN",
        a: "LEFT",
        d: "RIGHT",
      };
      const direction = map[e.key];
      if (direction && socketRef.current) {
        socketRef.current.emit("change_direction", { direction });
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  // REPAINT: initial paint + redraw on theme change (dark/light)
  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");

    if (!context) {
      console.warn("Canvas 2D context is not available");
      return;
    }

    draw();

    const observer = new MutationObserver(() => {
      draw();
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    return () => {
      observer.disconnect();
    };
  }, []); // redraw

  // RESIZE: fit board to viewport
  useEffect(() => {
    const handleResize = () => {
      const availW = window.innerWidth;
      const availH = window.innerHeight - HEADER_HEIGHT_PX;

      const { w, h } = gridRef.current;
      const newCell = Math.max(10, Math.floor(Math.min(availW / w, availH / h)));
      gridRef.current.cell = newCell;

      draw();
    };

    handleResize(); // fit once on load
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []); // resize

  return (
    <div className="absolute top-16 left-0 right-0 bottom-0 flex flex-col items-center justify-center">
      <canvas
        ref={canvasRef}
        width={gridRef.current.w * gridRef.current.cell}
        height={gridRef.current.h * gridRef.current.cell}
        style={{ position: "absolute", border: "none", outline: "none" }}
      />
      <div className="absolute rounded-lg p-8 w-fit flex flex-col items-center shadow-md backdrop-blur-md bg-background-trans">
        <span className="text-primary text-3xl font-extrabold mb-2 text-center">
          CSAI Student
        </span>
      </div>
    </div>
  );
}
