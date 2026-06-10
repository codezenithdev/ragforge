import { Loader2, Upload } from "lucide-react";
import { type DragEvent, useRef, useState } from "react";

import { useUploadDocument } from "@/hooks/useBrief";
import { cn } from "@/lib/utils";

const ACCEPTED = [".pdf", ".docx"];

export function UploadZone() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const upload = useUploadDocument();

  const handleFile = (file: File | undefined) => {
    setError(null);
    if (!file) return;
    const suffix = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPTED.includes(suffix)) {
      setError(`Unsupported type "${suffix}" — PDF or DOCX only.`);
      return;
    }
    upload.mutate(file, {
      onError: (e) => setError(e instanceof Error ? e.message : "Upload failed"),
    });
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    handleFile(event.dataTransfer.files[0]);
  };

  return (
    <div>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-6 text-center transition-colors",
          dragging
            ? "border-teal-500 bg-teal-50"
            : "border-slate-300 bg-white hover:border-teal-400 hover:bg-slate-50",
          upload.isPending && "pointer-events-none opacity-70",
        )}
        role="button"
        aria-label="Upload a PDF or DOCX document"
      >
        {upload.isPending ? (
          <>
            <Loader2 className="h-6 w-6 animate-spin text-teal-600" />
            <p className="text-sm font-medium text-slate-700">Uploading &amp; indexing…</p>
            <div className="h-1 w-40 overflow-hidden rounded-full bg-slate-200">
              <div className="h-full w-1/2 animate-pulse rounded-full bg-teal-500" />
            </div>
          </>
        ) : (
          <>
            <Upload className="h-6 w-6 text-slate-400" />
            <p className="text-sm font-medium text-slate-700">
              Drop a PDF or DOCX here, or click to browse
            </p>
            <p className="text-xs text-slate-500">
              Documents are chunked, embedded and indexed for retrieval
            </p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          className="hidden"
          onChange={(e) => {
            handleFile(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>
      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
      {upload.isSuccess && !upload.isPending && (
        <p className="mt-2 text-xs text-emerald-700">
          Indexed “{upload.data.name}” ({upload.data.num_chunks} chunks)
        </p>
      )}
    </div>
  );
}
