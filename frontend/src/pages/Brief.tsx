import { AlertTriangle, ArrowLeft, Loader2 } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { BriefViewer } from "@/components/BriefViewer";
import {
  BRIEF_POLL_HARD_CAP_MS,
  BRIEF_SLOW_THRESHOLD_MS,
  useBrief,
} from "@/hooks/useBrief";

export function BriefPage() {
  const { id } = useParams<{ id: string }>();
  const { data: brief, isLoading, isError } = useBrief(id ?? null);

  const active = brief?.status === "pending" || brief?.status === "processing";
  const elapsedMs = brief ? Date.now() - new Date(brief.created_at).getTime() : 0;
  const isSlow = active && elapsedMs > BRIEF_SLOW_THRESHOLD_MS;
  const isStalled = active && elapsedMs > BRIEF_POLL_HARD_CAP_MS;

  return (
    <div className="mx-auto max-w-4xl space-y-4 px-4 py-6">
      <Link
        to="/"
        className="inline-flex items-center gap-1 text-sm text-teal-700 hover:underline"
      >
        <ArrowLeft className="h-4 w-4" /> Back to workspace
      </Link>

      {isLoading && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading brief…
        </div>
      )}

      {isError && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          Brief not found.
        </div>
      )}

      {active && !isStalled && (
        <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-600">
          <Loader2 className="h-4 w-4 animate-spin text-teal-600" />
          Generating — current status: <span className="font-medium">{brief!.status}</span>
        </div>
      )}

      {isSlow && !isStalled && (
        <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-700">
          <AlertTriangle className="h-4 w-4" />
          This is taking longer than expected — still working…
        </div>
      )}

      {isStalled && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          <p className="font-medium">This brief seems to have stalled</p>
          <p className="mt-1">
            Generation hasn’t completed in time and polling has stopped. It will be marked failed
            automatically — check back shortly or start a new brief.
          </p>
        </div>
      )}

      {brief?.status === "failed" && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          <p className="font-medium">Brief generation failed</p>
          <p className="mt-1">{brief.result?.error ?? "Unknown error"}</p>
        </div>
      )}

      {brief?.status === "complete" && (
        <>
          <p className="text-sm italic text-slate-500">“{brief.query}”</p>
          <BriefViewer brief={brief} />
        </>
      )}
    </div>
  );
}
