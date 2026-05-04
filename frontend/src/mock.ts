/* Mock data shaped to match backend schemas — used when API is offline. */
import type { Document, Datapoint, VerbatimAnswer } from './types';

export const MOCK_DOCS: Document[] = [
  { id: 'asml-2025', source: 'asml.pdf', company: 'ASML', year: 2025, title: 'ASML Annual Report 2025', pages: 332, size_mb: 18.4, ingested_at: '2026-04-22T14:31:00Z', chunks: 4128, parser: 'docling', status: 'ready', fte: '44,209', net_zero_year: '2040' },
  { id: 'asml-2024', source: 'asml-2024.pdf', company: 'ASML', year: 2024, title: 'ASML Annual Report 2024', pages: 304, size_mb: 16.1, ingested_at: '2026-04-22T14:18:00Z', chunks: 3811, parser: 'docling', status: 'ready', fte: '41,697', net_zero_year: '2040' },
  { id: 'abn-amro-2025', source: 'abn-amro-2025.pdf', company: 'ABN AMRO', year: 2025, title: 'ABN AMRO Annual Report 2025', pages: 412, size_mb: 22.7, ingested_at: '2026-04-23T09:02:00Z', chunks: 5240, parser: 'llamaparse', status: 'ready', fte: '23,126', net_zero_year: '2050' },
  { id: 'shell-2025', source: 'shell-annual-report-2025.pdf', company: 'SHELL', year: 2025, title: 'Shell Annual Report 2025', pages: 446, size_mb: 28.9, ingested_at: '2026-04-23T11:47:00Z', chunks: 5872, parser: 'llamaparse', status: 'ready', fte: '~96,000', net_zero_year: '2050' },
  { id: 'heineken-2024', source: 'heineken-2024.pdf', company: 'HEINEKEN', year: 2024, title: 'Heineken Annual Report 2024', pages: 268, size_mb: 14.3, ingested_at: '2026-04-23T13:10:00Z', chunks: 3214, parser: 'docling', status: 'ready', fte: '85,870', net_zero_year: '2040' },
];

export const MOCK_DATAPOINTS: Datapoint[] = [
  { company: 'ASML', source: 'asml.pdf', page: 301, type: 'fte', metric: 'Total payroll & temporary employees (FTE, year-end)', value: '44,209', period: 2025, section: 'Note 18 — Employees', verbatim: 'Total payroll and temporary employees in FTE as of December 31, 2025: 44,209.' },
  { company: 'ASML', source: 'asml.pdf', page: 131, type: 'fte', metric: 'Average payroll employees (FTE)', value: '43,267', period: 2025, section: 'Remuneration / pay-ratio table', verbatim: 'Metric: Average number of payroll employees in FTEs — Period: 2025 — Value: 43,267.' },
  { company: 'ASML', source: 'asml.pdf', page: 131, type: 'fte', metric: 'Average payroll employees (FTE)', value: '41,697', period: 2024, section: 'Remuneration / pay-ratio table', verbatim: 'Metric: Average number of payroll employees in FTEs — Period: 2024 — Value: 41,697.' },
  { company: 'ASML', source: 'asml.pdf', page: 5, type: 'fte', metric: 'Total employees (highlight)', value: '> 44,000', period: 2025, section: 'At a glance', verbatim: 'Metric: Total employees (FTEs) — 2025 — > 44,000 (highlight).' },
  { company: 'ABN AMRO', source: 'abn-amro-2025.pdf', page: 47, type: 'fte', metric: 'Internal employees (FTE, year-end)', value: '23,126', period: 2025, section: 'Other information', verbatim: 'Total bank: Number of internal employees (end of period, in FTEs) — 2025: 23,126 — 2024: 21,976' },
  { company: 'ABN AMRO', source: 'abn-amro-2025.pdf', page: 47, type: 'fte', metric: 'External employees (FTE, year-end)', value: '2,216', period: 2025, section: 'Other information', verbatim: 'Total bank: Number of external employees (end of period, in FTEs) — 2025: 2,216 — 2024: 3,670' },
  { company: 'ABN AMRO', source: 'abn-amro-2025.pdf', page: 174, type: 'fte', metric: 'Net workforce reduction by 2028', value: '−5,200 FTE', period: '2024→2028', section: 'New strategy', verbatim: '…a net total reduction of the global workforce by 5,200 FTEs by 2028 compared with year-end 2024. Around half of the reductions are expected to take place through attrition.' },
  { company: 'SHELL', source: 'shell-annual-report-2025.pdf', page: 122, type: 'fte', metric: 'Total employees', value: '~96,000', period: 2025, section: 'People', verbatim: 'At the end of 2025, Shell employed approximately 96,000 people in more than 70 countries.' },
  { company: 'HEINEKEN', source: 'heineken-2024.pdf', page: 88, type: 'fte', metric: 'Total employees (FTE)', value: '85,870', period: 2024, section: 'Our people', verbatim: 'Total full-time equivalent employees as of 31 December 2024: 85,870.' },
  { company: 'ASML', source: 'asml.pdf', page: 178, type: 'esg', metric: 'GHG-neutral across value chain', value: 'by 2040', period: 'target', section: 'Climate Transition Plan', verbatim: 'We aim to be greenhouse gas neutral across our value chain by 2040, working in close partnership with customers and suppliers.' },
  { company: 'ASML', source: 'asml.pdf', page: 145, type: 'esg', metric: 'Net Scope 1 & 2 GHG-neutral', value: 'by 2025 — achieved', period: 'target', section: 'Our business strategy', verbatim: 'Target: Net scope 1 and 2 CO₂e emissions GHG neutral by 2025 — Performance: 0 kt' },
  { company: 'ASML', source: 'asml.pdf', page: 145, type: 'esg', metric: 'Gross Scope 1 & 2', value: '26 kt (target 45 kt)', period: 2025, section: 'Our business strategy', verbatim: 'Target: Gross scope 1 and 2 CO₂e emissions 45 kt by 2025 — Performance: 26 kt' },
  { company: 'ASML', source: 'asml.pdf', page: 145, type: 'esg', metric: 'Top-80% supplier commitment', value: '32% of 75% target', period: 'by 2026', section: 'Climate strategy', verbatim: 'Target: 75% commitment from top 80% suppliers by 2026 — Performance: 32%' },
  { company: 'ASML', source: 'asml.pdf', page: 6, type: 'esg', metric: 'Net Scope 3 emissions', value: '11.5 Mt', period: 2025, section: 'At a glance', verbatim: 'Target: Net scope 3 CO₂e emissions (Mt) GHG neutral by 2040 — Performance: 11.5 Mt' },
  { company: 'ABN AMRO', source: 'abn-amro-2025.pdf', page: 25, type: 'esg', metric: 'Net-zero economy', value: 'by 2050', period: 'target', section: 'Sustainability pillar', verbatim: 'Our climate strategy supports the transition to a net-zero economy by 2050 by aligning our portfolio with the Paris Agreement.' },
  { company: 'ABN AMRO', source: 'abn-amro-2025.pdf', page: 19, type: 'esg', metric: 'Climate strategy alignment', value: 'well-below-2°C', period: 'announced Nov 2025', section: 'Help clients achieve climate targets', verbatim: 'In November 2025, we announced our intention to update our climate strategy to align with a well-below-2°C pathway.' },
  { company: 'ABN AMRO', source: 'abn-amro-2025.pdf', page: 31, type: 'esg', metric: 'Office energy use', value: '93 kWh/m²', period: 2025, section: 'Own operations', verbatim: 'Office energy use dropped from 184 kWh/m² in 2018 to 93 kWh/m² in 2025.' },
  { company: 'SHELL', source: 'shell-annual-report-2025.pdf', page: 5, type: 'esg', metric: 'Net-zero energy business', value: 'by 2050', period: 'target', section: 'Chair\u2019s message', verbatim: 'We have a target to become a net-zero emissions energy business by 2050.' },
  { company: 'SHELL', source: 'shell-annual-report-2025.pdf', page: 13, type: 'esg', metric: 'Halve Scope 1 & 2 emissions', value: '~70% of target by 2025', period: 'target 2030', section: 'Our strategy', verbatim: 'As of 2025, we have already achieved some 70% of that target.' },
  { company: 'SHELL', source: 'shell-annual-report-2025.pdf', page: 90, type: 'esg', metric: 'Total Scope 1+2 (net)', value: '53 Mt CO₂e', period: 2025, section: 'Less emissions', verbatim: 'In 2025, total combined Scope 1 and 2 GHG emissions (net) were 53 million tonnes of CO₂e.' },
  { company: 'SHELL', source: 'shell-annual-report-2025.pdf', page: 13, type: 'esg', metric: 'Methane intensity', value: '< 0.2%', period: 'target', section: 'Less emissions', verbatim: 'Maintain methane emissions intensity below 0.2% and achieve near-zero methane emissions intensity by 2030.' },
];

export const SAMPLE_ANSWERS: Record<string, VerbatimAnswer & { caveat?: string }> = {
  asml_fte: {
    question: 'How many total employees did ASML have at the end of 2025?',
    answer: 'ASML reported 44,209 total payroll and temporary employees (in FTE) as of December 31, 2025.',
    verbatim: 'Total payroll and temporary employees in FTE as of December 31, 2025: 44,209.',
    citations: [
      { source: 'asml.pdf', page: 301, quote: 'Total payroll and temporary employees in FTE as of December 31, 2025: 44,209.' },
      { source: 'asml.pdf', page: 5, quote: 'Total employees (FTEs) — 2025 — > 44,000 (highlight).' },
    ],
    refused: false,
    refusal_reason: null,
    raw_citations: [],
    grounding_drops: [],
  },
  shell_climate: {
    question: 'How much did SHELL spend on climate change adaption in 2025?',
    answer: 'Shell does not publish a single line item labelled "climate change adaptation" in 2025. The closest disclosed figure is R&D on decarbonisation projects: ~$485 million (about 41% of total R&D spend).',
    verbatim: 'In 2025, our R&D expenditure on projects that aim to contribute to decarbonisation was around $485 million, representing about 41% of our total R&D spend, compared with around 45% in 2024.',
    citations: [
      { source: 'shell-annual-report-2025.pdf', page: 87, quote: 'In 2025, our R&D expenditure on projects that aim to contribute to decarbonisation was around $485 million, representing about 41% of our total R&D spend.' },
      { source: 'shell-annual-report-2025.pdf', page: 13, quote: 'We plan cash capital expenditure within the range of $20–22 billion per year between 2025 and 2028, with $21 billion in 2025.' },
    ],
    refused: false,
    refusal_reason: null,
    raw_citations: [],
    grounding_drops: [],
    caveat: 'No verbatim "climate adaptation spend" exists in the report — figure shown is decarbonisation R&D, the nearest reported proxy.',
  },
  abn_workforce: {
    question: 'What is ABN AMRO\u2019s plan for workforce reduction?',
    answer: 'ABN AMRO announced in November 2025 a net reduction of 5,200 FTE by 2028 compared with year-end 2024. About half of the reductions are expected to occur through attrition. By end of 2025, around 1,500 FTE had already been reduced.',
    verbatim: '…a net total reduction of the global workforce by 5,200 FTEs by 2028 compared with year-end 2024. Around half of the reductions are expected to take place through attrition. In 2025, a reduction of approximately 1,500 FTEs was realised.',
    citations: [
      { source: 'abn-amro-2025.pdf', page: 174, quote: 'ABN AMRO announced it would rightsize the organisation, resulting in a net total reduction of the global workforce by 5,200 FTEs by 2028 compared with year-end 2024.' },
      { source: 'abn-amro-2025.pdf', page: 15, quote: 'The plan includes a net reduction of the workforce by 5,200 full-time equivalents (FTEs) by 2028 compared with 2024, with half of the reductions expected to occur through attrition.' },
    ],
    refused: false,
    refusal_reason: null,
    raw_citations: [],
    grounding_drops: [],
  },
  refusal: {
    question: 'How many company cars did ASML own in 2025?',
    answer: '',
    verbatim: null,
    refused: true,
    refusal_reason: 'Number of company-owned vehicles is not disclosed in the ASML 2025 annual report. The report covers fleet-related emissions intensity but not a vehicle count.',
    citations: [],
    raw_citations: [],
    grounding_drops: [],
  },
};

export const SUGGESTED_QUESTIONS = [
  'How many total employees did ASML have at the end of 2025?',
  'How much did SHELL spend on climate change adaption in 2025?',
  'What is ABN AMRO\u2019s plan for workforce reduction?',
  'When does ASML aim to be GHG neutral across its value chain?',
  'What were Shell\u2019s total Scope 1 and 2 emissions in 2025?',
  'How many company cars did ASML own in 2025?',
];
