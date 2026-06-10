import { type HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

type Variant = "default" | "secondary" | "success" | "warning" | "destructive";

const variantClasses: Record<Variant, string> = {
  default: "bg-teal-100 text-teal-800",
  secondary: "bg-slate-100 text-slate-700",
  success: "bg-emerald-100 text-emerald-800",
  warning: "bg-amber-100 text-amber-800",
  destructive: "bg-red-100 text-red-800",
};

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: Variant;
}

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium",
        variantClasses[variant],
        className,
      )}
      {...props}
    />
  );
}
