import { cn } from "@/lib/utils";

/**
 * Visual 0-100% faithfulness bar.
 * Green > 0.8, yellow 0.6-0.8, red < 0.6 (matching the backend warn threshold).
 */
export function ConfidenceBar({ score, className }: { score: number; className?: string }) {
  const pct = Math.round(Math.max(0, Math.min(1, score)) * 100);
  const fill =
    score > 0.8 ? "bg-emerald-500" : score >= 0.6 ? "bg-amber-500" : "bg-red-500";
  const label =
    score > 0.8 ? "text-emerald-700" : score >= 0.6 ? "text-amber-700" : "text-red-700";

  return (
    <div className={cn("flex items-center gap-2", className)} title={`Faithfulness: ${pct}%`}>
      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-200">
        <div
          className={cn("h-full rounded-full transition-all", fill)}
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
      <span className={cn("text-xs font-semibold tabular-nums", label)}>{pct}%</span>
    </div>
  );
}
