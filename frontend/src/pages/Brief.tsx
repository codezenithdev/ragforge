import { ArrowLeft, Loader2 } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { BriefViewer } from "@/components/BriefViewer";
import { useBrief } from "@/hooks/useBrief";

export function BriefPage() {
  const { id } = useParams<{ id: string }>();
  const { data: brief, isLoading, isError } = useBrief(id ?? null);

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

      {brief && (brief.status === "pending" || brief.status === "processing") && (
        <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-600">
          <Loader2 className="h-4 w-4 animate-spin text-teal-600" />
          Generating — current status: <span className="font-medium">{brief.status}</span>
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
