import { AlertTriangle, ExternalLink, Globe, Loader2 } from "lucide-react";

import { ConfidenceBar } from "@/components/ConfidenceBar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useRunEval } from "@/hooks/useBrief";
import {
  type Brief,
  type BriefSection,
  type SourceReference,
} from "@/lib/api";

const LOW_CONFIDENCE = 0.6; // mirrors settings.faithfulness_warn_threshold

function Citations({
  sectionSources,
  allSources,
}: {
  sectionSources: string[];
  allSources: SourceReference[];
}) {
  const refs = sectionSources
    .map((id) => allSources.find((s) => s.id === id))
    .filter((s): s is SourceReference => s !== undefined);
  if (refs.length === 0) return null;

  return (
    <details className="mt-2">
      <summary className="cursor-pointer text-xs font-medium text-teal-700 hover:underline">
        {refs.length} source{refs.length > 1 ? "s" : ""} cited
      </summary>
      <ul className="mt-1 space-y-1 border-l-2 border-slate-200 pl-3">
        {refs.map((ref) => (
          <li key={ref.id} className="flex items-center gap-1.5 text-xs text-slate-600">
            <span className="font-mono text-slate-400">[{ref.id}]</span>
            {ref.source_type === "web" && <Globe className="h-3 w-3 text-sky-500" />}
            {ref.url ? (
              <a
                href={ref.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-teal-700 hover:underline"
              >
                {ref.title ?? ref.url}
                <ExternalLink className="h-3 w-3" />
              </a>
            ) : (
              <span>{ref.title ?? "document chunk"}</span>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}

function Section({
  title,
  section,
  allSources,
}: {
  title: string;
  section: BriefSection;
  allSources: SourceReference[];
}) {
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle className="flex items-center gap-2">
          {title}
          {section.confidence < LOW_CONFIDENCE && (
            <Badge variant="warning">
              <AlertTriangle className="h-3 w-3" /> Low confidence
            </Badge>
          )}
        </CardTitle>
        <ConfidenceBar score={section.confidence} />
      </CardHeader>
      <CardContent>
        <p className="text-sm leading-relaxed text-slate-700">{section.content}</p>
        <Citations sectionSources={section.sources} allSources={allSources} />
      </CardContent>
    </Card>
  );
}

export function BriefViewer({ brief }: { brief: Brief }) {
  const runEval = useRunEval();
  const result = brief.result;
  if (!result) return null;
  const ragas = result.ragas_eval;

  return (
    <div className="space-y-4">
      <div>
        <h2 className="font-display text-2xl font-semibold text-slate-900">{result.title}</h2>
        <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
          {result.generated_at && (
            <span>Generated {new Date(result.generated_at).toLocaleString()}</span>
          )}
          {result.crag_action === "corrective_search" && (
            <Badge variant="secondary">
              <Globe className="h-3 w-3" /> Web-corrected retrieval
            </Badge>
          )}
        </div>
      </div>

      <Section
        title="Executive summary"
        section={result.executive_summary}
        allSources={result.sources}
      />

      <div className="space-y-2">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Key facts
        </h3>
        {result.key_facts.map((fact, i) => (
          <Section
            key={i}
            title={`Key fact ${i + 1}`}
            section={fact}
            allSources={result.sources}
          />
        ))}
      </div>

      <Section
        title="Risks & limitations"
        section={result.risks_and_limitations}
        allSources={result.sources}
      />
      <Section
        title="Opportunities"
        section={result.opportunities}
        allSources={result.sources}
      />

      {result.open_questions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Open questions the sources could not answer</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">
              {result.open_questions.map((q) => (
                <li key={q}>{q}</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle>RAGAS evaluation</CardTitle>
          <Button
            size="sm"
            variant="outline"
            onClick={() => runEval.mutate(brief.brief_id)}
            disabled={runEval.isPending}
          >
            {runEval.isPending && <Loader2 className="h-3 w-3 animate-spin" />}
            {ragas ? "Re-run" : "Run evaluation"}
          </Button>
        </CardHeader>
        <CardContent>
          {ragas ? (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {(
                [
                  ["Faithfulness", ragas.faithfulness],
                  ["Relevancy", ragas.answer_relevancy],
                  ["Precision", ragas.context_precision],
                  ["Overall", ragas.overall],
                ] as const
              ).map(([label, value]) => (
                <div key={label} className="rounded-md bg-slate-50 p-2 text-center">
                  <p className="text-lg font-semibold tabular-nums text-slate-900">
                    {value === null || value === undefined ? "—" : value.toFixed(2)}
                  </p>
                  <p className="text-xs text-slate-500">{label}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-500">
              Score this brief with RAGAS (faithfulness, relevancy, context precision).
              {runEval.isPending && " Running — this takes ~30s…"}
            </p>
          )}
          {runEval.isError && (
            <p className="mt-2 text-xs text-red-600">
              {runEval.error instanceof Error ? runEval.error.message : "Evaluation failed"}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
