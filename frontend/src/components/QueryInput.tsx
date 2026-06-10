import { CheckCircle2, Loader2, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { useCreateBrief } from "@/hooks/useBrief";
import { type BriefStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * Pipeline stages with rough durations (seconds). The backend exposes only
 * coarse status (pending/processing/complete), so the per-stage progress is a
 * timed estimate that mirrors the real node order in the LangGraph pipeline.
 */
const PIPELINE_STAGES: Array<[label: string, seconds: number]> = [
  ["Decomposing query into sub-questions", 8],
  ["Writing HyDE retrieval passages", 8],
  ["Retrieving sources (hybrid + RRF)", 7],
  ["Evaluating retrieval quality (CRAG)", 5],
  ["Re-ranking with cross-encoder", 4],
  ["Generating the brief", 16],
  ["Scoring per-section faithfulness", 600],
];

function PipelineProgress({ startedAt }: { startedAt: number }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 1000);
    return () => clearInterval(timer);
  }, [startedAt]);

  let cumulative = 0;
  let currentIndex = PIPELINE_STAGES.length - 1;
  for (let i = 0; i < PIPELINE_STAGES.length; i++) {
    cumulative += PIPELINE_STAGES[i][1];
    if (elapsed < cumulative) {
      currentIndex = i;
      break;
    }
  }

  return (
    <div className="space-y-1.5 rounded-lg border border-slate-200 bg-white p-4">
      {PIPELINE_STAGES.map(([label], i) => (
        <div
          key={label}
          className={cn(
            "flex items-center gap-2 text-sm",
            i < currentIndex && "text-emerald-700",
            i === currentIndex && "font-medium text-slate-900",
            i > currentIndex && "text-slate-400",
          )}
        >
          {i < currentIndex ? (
            <CheckCircle2 className="h-4 w-4 shrink-0" />
          ) : i === currentIndex ? (
            <Loader2 className="h-4 w-4 shrink-0 animate-spin text-teal-600" />
          ) : (
            <span className="inline-block h-4 w-4 shrink-0 rounded-full border border-slate-300" />
          )}
          {label}
        </div>
      ))}
      <p className="pt-1 text-xs text-slate-400">{Math.round(elapsed)}s elapsed</p>
    </div>
  );
}

interface QueryInputProps {
  selectedDocIds: string[];
  activeStatus: BriefStatus | null;
  onCreated: (briefId: string) => void;
}

export function QueryInput({ selectedDocIds, activeStatus, onCreated }: QueryInputProps) {
  const [query, setQuery] = useState("");
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const create = useCreateBrief();
  const running = activeStatus === "pending" || activeStatus === "processing";

  const submit = () => {
    if (query.trim().length < 3 || running) return;
    setStartedAt(Date.now());
    create.mutate(
      { query: query.trim(), documentIds: selectedDocIds },
      { onSuccess: (data) => onCreated(data.brief_id) },
    );
  };

  return (
    <div className="space-y-3">
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder='Ask a research question, e.g. "What is Anthropic&apos;s competitive position in the LLM market?"'
        rows={3}
        className="w-full resize-none rounded-lg border border-slate-300 bg-white p-3 text-sm shadow-sm focus:border-teal-500 focus:outline-none focus:ring-2 focus:ring-teal-200"
      />
      <div className="flex items-center justify-between">
        <p className="text-xs text-slate-500">
          {selectedDocIds.length > 0
            ? `Scoped to ${selectedDocIds.length} selected document${selectedDocIds.length > 1 ? "s" : ""}`
            : "Searching all uploaded documents"}
        </p>
        <Button onClick={submit} disabled={query.trim().length < 3 || running || create.isPending}>
          {running || create.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Sparkles className="h-4 w-4" />
          )}
          Generate Brief
        </Button>
      </div>
      {create.isError && (
        <p className="text-xs text-red-600">
          {create.error instanceof Error ? create.error.message : "Failed to create brief"}
        </p>
      )}
      {running && startedAt !== null && <PipelineProgress startedAt={startedAt} />}
    </div>
  );
}
