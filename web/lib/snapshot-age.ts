export const STALE_SNAPSHOT_MAX_DAYS = 35;

const MS_PER_DAY = 24 * 60 * 60 * 1000;

function parseIsoDate(value: string): { year: number; month: number; day: number } | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return null;
  const [year, month, day] = match.slice(1).map(Number);
  const date = new Date(Date.UTC(year, month - 1, day));
  if (date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day) return null;
  return { year, month, day };
}

// Devolve "dados de DD/MM" quando a data de corte tem mais de 35 dias em relação a `today` (data local do navegador);
// null quando está dentro do prazo ou não é uma data ISO (ex.: "unknown").
export function staleSnapshotLabel(cutoff: string, today: Date): string | null {
  const parsed = parseIsoDate(cutoff);
  if (!parsed) return null;
  const cutoffDay = Date.UTC(parsed.year, parsed.month - 1, parsed.day);
  const todayDay = Date.UTC(today.getFullYear(), today.getMonth(), today.getDate());
  if ((todayDay - cutoffDay) / MS_PER_DAY <= STALE_SNAPSHOT_MAX_DAYS) return null;
  const pad = (value: number) => String(value).padStart(2, "0");
  return `dados de ${pad(parsed.day)}/${pad(parsed.month)}`;
}
