export function reportLabel(doc: { company: string | null; year: number | null; source: string }): string {
  const parts = [doc.company, doc.year ? String(doc.year) : null].filter(Boolean);
  return parts.length ? parts.join(' ') : doc.source;
}
