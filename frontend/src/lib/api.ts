/**
 * Typed client for the Briefr API. Shapes mirror the backend's Pydantic
 * schemas (BriefOutput etc.) and route serializers.
 */

const API_URL: string =
  (import.meta.env.VITE_API_URL as string | undefined) ??
  "http://localhost:8000/api/v1";

// Sent as the X-API-Key header on every request when configured (P0.1). Unset
// in local dev, where the backend leaves auth disabled.
const API_KEY: string | undefined = import.meta.env.VITE_API_KEY as
  | string
  | undefined;

/** Merge the configured API key header into a request's headers, if any. */
function withAuth(headers?: HeadersInit): HeadersInit | undefined {
  if (!API_KEY) return headers;
  return { ...(headers ?? {}), "X-API-Key": API_KEY };
}

export type BriefStatus = "pending" | "processing" | "complete" | "failed";

export interface DocumentInfo {
  document_id: string;
  name: string;
  source_type: string;
  created_at: string;
}

export interface BriefSection {
  content: string;
  sources: string[];
  confidence: number;
}

export interface SourceReference {
  id: string;
  source_type: "document" | "web";
  title: string | null;
  url: string | null;
}

export interface RagasEval {
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
  context_recall: number | null;
  overall: number;
  raw: Record<string, number>;
}

export interface BriefResult {
  title: string;
  executive_summary: BriefSection;
  key_facts: BriefSection[];
  risks_and_limitations: BriefSection;
  opportunities: BriefSection;
  open_questions: string[];
  sources: SourceReference[];
  generated_at: string | null;
  contexts?: string[];
  crag_action?: string;
  ragas_eval?: RagasEval;
  /** Only present when status === "failed". */
  error?: string;
}

export interface Brief {
  brief_id: string;
  query: string;
  status: BriefStatus;
  result: BriefResult | null;
  faithfulness_scores: Record<string, number> | null;
  created_at: string;
  completed_at: string | null;
}

export interface BriefListItem {
  brief_id: string;
  query: string;
  status: BriefStatus;
  created_at: string;
  completed_at: string | null;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: withAuth(init?.headers),
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // non-JSON error body; keep statusText
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export async function uploadDocument(file: File): Promise<DocumentInfo & { num_chunks: number }> {
  const form = new FormData();
  form.append("file", file);
  return request("/documents/upload", { method: "POST", body: form });
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  return request("/documents");
}

export async function deleteDocument(documentId: string): Promise<void> {
  return request(`/documents/${documentId}`, { method: "DELETE" });
}

export async function createBrief(
  query: string,
  documentIds: string[],
): Promise<{ brief_id: string; status: BriefStatus }> {
  return request("/briefs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, document_ids: documentIds }),
  });
}

export async function getBrief(briefId: string): Promise<Brief> {
  return request(`/briefs/${briefId}`);
}

export async function listBriefs(): Promise<BriefListItem[]> {
  return request("/briefs");
}

export async function runEval(briefId: string): Promise<RagasEval & { brief_id: string }> {
  return request("/eval/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ brief_id: briefId }),
  });
}

/** Poll a brief every `intervalMs` until it completes or fails. */
export async function pollBrief(
  briefId: string,
  onUpdate: (brief: Brief) => void,
  intervalMs = 2000,
): Promise<Brief> {
  for (;;) {
    const brief = await getBrief(briefId);
    onUpdate(brief);
    if (brief.status === "complete" || brief.status === "failed") return brief;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
