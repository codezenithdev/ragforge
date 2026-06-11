import { FileText, Loader2, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { BriefViewer } from "@/components/BriefViewer";
import { QueryInput } from "@/components/QueryInput";
import { UploadZone } from "@/components/UploadZone";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useBrief, useBriefs, useDeleteDocument, useDocuments } from "@/hooks/useBrief";
import { type BriefStatus } from "@/lib/api";

const STATUS_VARIANT: Record<BriefStatus, "secondary" | "default" | "success" | "destructive"> = {
  pending: "secondary",
  processing: "default",
  complete: "success",
  failed: "destructive",
};

function DocumentList({
  selected,
  onToggle,
}: {
  selected: string[];
  onToggle: (id: string) => void;
}) {
  const { data: documents, isLoading } = useDocuments();
  const deleteDoc = useDeleteDocument();

  if (isLoading) return <p className="text-xs text-slate-500">Loading documents…</p>;
  if (!documents || documents.length === 0)
    return <p className="text-xs text-slate-500">No documents yet — upload one above.</p>;

  return (
    <ul className="space-y-1">
      {documents.map((doc) => (
        <li
          key={doc.document_id}
          className="flex items-center gap-2 rounded-md border border-slate-200 bg-white px-2 py-1.5"
        >
          <input
            type="checkbox"
            id={`doc-${doc.document_id}`}
            checked={selected.includes(doc.document_id)}
            onChange={() => onToggle(doc.document_id)}
            disabled={doc.status !== undefined && doc.status !== "ready"}
            className="h-4 w-4 accent-teal-700 disabled:opacity-40"
          />
          <FileText className="h-4 w-4 shrink-0 text-slate-400" />
          <label
            htmlFor={`doc-${doc.document_id}`}
            className="min-w-0 flex-1 cursor-pointer truncate text-sm text-slate-700"
            title={doc.name}
          >
            {doc.name}
          </label>
          {doc.status === "pending" || doc.status === "processing" ? (
            <span className="shrink-0 text-xs text-slate-400">indexing…</span>
          ) : doc.status === "failed" ? (
            <span className="shrink-0 text-xs text-red-600" title={doc.error ?? undefined}>
              failed
            </span>
          ) : null}
          <Button
            size="icon"
            variant="destructive"
            aria-label={`Delete ${doc.name}`}
            onClick={() => deleteDoc.mutate(doc.document_id)}
            disabled={deleteDoc.isPending}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </li>
      ))}
    </ul>
  );
}

function RecentBriefs() {
  const { data: briefs } = useBriefs();
  if (!briefs || briefs.length === 0) return null;

  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        Recent briefs
      </h3>
      <ul className="space-y-1">
        {briefs.slice(0, 8).map((brief) => (
          <li key={brief.brief_id}>
            <Link
              to={`/briefs/${brief.brief_id}`}
              className="flex items-center justify-between gap-2 rounded-md px-2 py-1.5 text-sm text-slate-700 hover:bg-slate-100"
            >
              <span className="min-w-0 truncate" title={brief.query}>
                {brief.query}
              </span>
              <Badge variant={STATUS_VARIANT[brief.status]}>{brief.status}</Badge>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function Home() {
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [activeBriefId, setActiveBriefId] = useState<string | null>(null);
  const { data: activeBrief } = useBrief(activeBriefId);

  const toggleDoc = (id: string) =>
    setSelectedDocIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );

  return (
    <div className="mx-auto grid max-w-6xl gap-6 px-4 py-6 lg:grid-cols-[320px_1fr]">
      <aside className="space-y-5">
        <UploadZone />
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Your documents
          </h3>
          <DocumentList selected={selectedDocIds} onToggle={toggleDoc} />
        </div>
        <RecentBriefs />
      </aside>

      <main className="space-y-5">
        <QueryInput
          selectedDocIds={selectedDocIds}
          activeStatus={activeBrief?.status ?? null}
          onCreated={setActiveBriefId}
        />

        {activeBrief?.status === "complete" && <BriefViewer brief={activeBrief} />}

        {activeBrief?.status === "failed" && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            <p className="font-medium">Brief generation failed</p>
            <p className="mt-1">{activeBrief.result?.error ?? "Unknown error"}</p>
          </div>
        )}

        {!activeBrief && (
          <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-slate-300 p-12 text-center">
            <Loader2 className="h-5 w-5 text-slate-300" />
            <p className="text-sm text-slate-500">
              Upload reference documents, then ask a question to generate a sourced brief
              with per-section faithfulness scores.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
