"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";

import { Button } from "@/components/ui/button";

export function ThemeButton() {
  const { theme, setTheme } = useTheme();

  return (
    <div className="flex items-center sm:inline-flex sm:w-auto w-full justify-center">
      <Button
        variant="ghost"
        size="icon"
        className="transition-all p-3 sm:p-2 flex items-center justify-center shadow-sm border border-border w-full sm:w-auto"
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        aria-label="Toggle light/dark mode"
      >
        {theme === "light" ? (
          <Moon className="h-6 w-6 sm:h-[1.2rem] sm:w-[1.2rem]" />
        ) : (
          <Sun className="h-6 w-6 sm:h-[1.2rem] sm:w-[1.2rem]" />
        )}
        <span className="sm:hidden">
          {theme === "light" ? "light" : "dark"}
        </span>
      </Button>
    </div>
  );
}
