# Snake Bootcamp - Frontend Guide

Welcome to the **CSAI Snake Bootcamp Frontend**! This is a **guided learning project** where you'll build a React-based frontend to visualize an AI-powered Snake game in real-time.

## Table of Contents

- [What You'll Learn](#what-youll-learn)
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [What You Need to Implement](#what-you-need-to-implement)
- [Key Technologies You'll Use](#key-technologies-youll-use)
- [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
- [Design Guidelines](#design-guidelines)
- [Common Issues & Solutions](#common-issues--solutions)
- [Going Further (Optional Enhancements)](#going-further-optional-enhancements)
- [Tips for Success](#tips-for-success)
- [Resources](#resources)
- [Success Criteria](#success-criteria)

## What You'll Learn

This project will teach you:

- **Real-time web communication** with WebSockets
- **React development** with Next.js and TypeScript
- **Canvas rendering** for game visualization
- **State management** in React applications
- **Responsive design** and modern UI development

## Project Overview

You're building a **visual interface** that connects to a Python backend running a Snake game with AI agents. The frontend displays the game in real-time, showing the snake learning and improving its gameplay.

**This is NOT a build-from-scratch project** - we've provided the structure and guidance. Your job is to **fill in the implementation** where you see `TODO` comments!

## Project Structure

Your frontend has this structure:

    ```
    frontend/
    ├── app/
    │   ├── page.tsx              # Main game page (has TODOs for you!)
    │   └── layout.tsx            # App layout
    ├── components/
    │   ├── Header.tsx            # Navigation component
    │   └── ui/                   # Pre-built UI components
    └── lib/
        └── utils.ts              # Utility functions
    ```

## Getting Started

### 1. Install Dependencies

The project is already set up with Next.js and TypeScript. Install everything you need:

    ```bash
    cd apps/frontend
    npm install
    ```

### 2. Start the Development Server

    ```bash
    npm run dev
    ```

Your frontend will run at `http://localhost:3000`

### 3. Find the TODOs

Open `app/page.tsx` - this is where you'll implement the main functionality! Look for comments starting with `TODO:` - these guide you through what to implement.

## What You Need to Implement

The main file you'll work with is `app/page.tsx`. Here's what you need to complete:

### 1. **WebSocket Connection**

- Connect to the backend server at `localhost:8765`
- Handle connection events and game updates
- Send game initialization data

### 2. **Game State Management**

- Track snake position, food location, and score
- Update state based on server messages
- Handle game over scenarios

### 3. **Canvas Rendering**

- Draw the game grid
- Render the snake and food
- Update the display in real-time
- Make it look awesome!

### 4. **Responsive Design**

- Handle different screen sizes
- Manage canvas dimensions
- Ensure the game looks good everywhere

## Key Technologies You'll Use

### Socket.IO Client

    ```typescript
    import { io, Socket } from "socket.io-client";

    // You'll use this to connect to the backend
    const socket = io("localhost:8765");
    ```

### HTML5 Canvas

    ```typescript
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");

    // You'll use this to draw the game
    ```

### React Hooks

    ```typescript
    import { useEffect, useRef, useState } from "react";

    // You'll use these to manage state and side effects
    ```

## Step-by-Step Implementation Guide

### Step 1: WebSocket Setup

1. Look at the `useEffect` in `app/page.tsx`
2. Uncomment and implement the socket connection
3. Add event handlers for `connect` and `update` events
4. Test the connection with the backend

### Step 2: Game State Variables

1. Add state variables for snake, food, score, etc.
2. Update these variables when receiving server data
3. Make sure your state structure matches the server data

### Step 3: Canvas Drawing

1. Implement the drawing function to render the game
2. Draw the grid, snake, and food
3. Make sure coordinates match the server data
4. Add some styling to make it look nice!

### Step 4: Real-time Updates

1. Connect your drawing function to state changes
2. Ensure smooth updates when new data arrives
3. Handle edge cases like disconnections

### Step 5: Polish and Personalization

1. Add your personal branding/text
2. Improve the visual design
3. Add responsive features
4. Test on different devices

## Design Guidelines

### Colors and Theme

- The project uses a **light/dark theme system**
- Use the existing theme variables in your CSS
- Keep the design **clean and minimal**
- Focus on the **game visualization**

### Canvas Styling

- Make grid cells clearly visible
- Use distinct colors for snake vs food
- Consider adding smooth animations
- Ensure good contrast in both themes

### Responsive Design

- Test on mobile and desktop
- Adjust canvas size based on viewport
- Keep the game playable on all screen sizes

## Common Issues & Solutions

### WebSocket Won't Connect

- Make sure the backend server is running (`cd apps/backend && python src/app.py`)
- Check the WebSocket URL matches your backend port
- Look at browser console for connection errors

### Canvas Not Updating

- Ensure your drawing function is called when state changes
- Check that coordinates are within canvas bounds
- Verify the game data structure matches your expectations

### State Not Updating

- Make sure you're updating state in the WebSocket event handlers
- Check that your useEffect dependencies are correct
- Use React DevTools to inspect state changes

## Going Further (Optional Enhancements)

Once you complete the basic requirements, try these challenges:

### Beginner Enhancements

- Add game statistics (games played, best score)
- Implement smooth animations between moves
- Add sound effects for eating food

### Intermediate Enhancements

- Create a game history viewer
- Add different visual themes for the snake
- Implement game speed controls

### Advanced Enhancements

- Add multiple AI agents competing
- Create data visualizations for AI performance
- Build a replay system for interesting games

## Tips for Success

1. **Start Small**: Get the basic connection working first
2. **Use Console Logs**: Debug your WebSocket data with `console.log()`
3. **Test Frequently**: Check your progress after each small change
4. **Ask for Help**: Don't hesitate to ask club members for assistance
5. **Have Fun**: This is your chance to be creative!

## Resources

- [Socket.IO Client Docs](https://socket.io/docs/v4/client-api/)
- [Canvas API Reference](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)
- [React Hooks Guide](https://react.dev/reference/react)
- [Next.js Documentation](https://nextjs.org/docs)

## Success Criteria

Your frontend is complete when:

- WebSocket connects to backend successfully
- Game displays in real-time with smooth updates
- Snake and food render correctly on canvas
- Responsive design works on mobile and desktop
- Code is clean and well-commented
- You've added your personal creative touches!

---

**Remember**: This is **your project** to showcase your skills! While we provide the structure, feel free to add your own creative flair and improvements. The goal is to learn while building something awesome that represents your work to the CSAI community.

**Need help?** Check the existing code, ask club members, or refer to the documentation links above. You've got this!
