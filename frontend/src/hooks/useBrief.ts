import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import {
  createBrief,
  deleteDocument,
  getBrief,
  listBriefs,
  listDocuments,
  runEval,
  uploadDocument,
} from "@/lib/api";

export function useDocuments() {
  return useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
    // Poll while any document is still ingesting so pending -> ready reflects.
    refetchInterval: (query) =>
      query.state.data?.some((d) => d.status === "pending" || d.status === "processing")
        ? 2000
        : false,
  });
}

export function useUploadDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: uploadDocument,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["documents"] }),
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["documents"] }),
  });
}

/**
 * Hard ceiling on client-side polling (P1.8). The backend sweeper fails truly
 * stuck briefs within ~20 min, so this is defense-in-depth: stop polling rather
 * than hammer the API forever if the backend never updates the status.
 */
export const BRIEF_POLL_HARD_CAP_MS = 25 * 60 * 1000;
/** After this long in pending/processing, the UI warns it's taking longer than expected. */
export const BRIEF_SLOW_THRESHOLD_MS = 90 * 1000;

/** Fetch a brief, polling every 2s while it is still pending/processing (capped). */
export function useBrief(briefId: string | null) {
  return useQuery({
    queryKey: ["brief", briefId],
    queryFn: () => getBrief(briefId!),
    enabled: briefId !== null,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 2000;
      const active = data.status === "pending" || data.status === "processing";
      if (!active) return false;
      const elapsed = Date.now() - new Date(data.created_at).getTime();
      return elapsed > BRIEF_POLL_HARD_CAP_MS ? false : 2000;
    },
  });
}

export function useBriefs() {
  return useQuery({ queryKey: ["briefs"], queryFn: listBriefs });
}

export function useCreateBrief() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ query, documentIds }: { query: string; documentIds: string[] }) =>
      createBrief(query, documentIds),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["briefs"] }),
  });
}

export function useRunEval() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: runEval,
    onSuccess: (_data, briefId) =>
      queryClient.invalidateQueries({ queryKey: ["brief", briefId] }),
  });
}
