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
  return useQuery({ queryKey: ["documents"], queryFn: listDocuments });
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

/** Fetch a brief, polling every 2s while it is still pending/processing. */
export function useBrief(briefId: string | null) {
  return useQuery({
    queryKey: ["brief", briefId],
    queryFn: () => getBrief(briefId!),
    enabled: briefId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "pending" || status === "processing" ? 2000 : false;
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
