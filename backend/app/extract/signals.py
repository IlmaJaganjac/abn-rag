from __future__ import annotations

import re

PUNCT_RE = re.compile(r"[^\w\s]")
SPACE_RE = re.compile(r"\s+")
SYMBOL_SPACE_RE = re.compile(r"\s*([€$£%><=,.])\s*")
PLACEHOLDER_VALUE = re.compile(
    r"^(?:unknown|n/?a|none|null|[-—]+|[€$£]?\s*x+x+%?|x+x+%?)$",
    re.IGNORECASE,
)
STATUS_ONLY_VALUE = re.compile(
    r"^(?:approved|executed|suspended|modified|discontinued|may\s+(?:suspend|not declare|pay).*|"
    r"on\s+track|tbc|tbd|in\s+progress|ongoing)$",
    re.IGNORECASE,
)
ACTUAL_METRIC = re.compile(r"\bactual\b|\breported\b|\bperformance\b", re.IGNORECASE)
PROGRESS_SIGNAL = re.compile(
    r"\bprogress\b|achieved|completion|against\s+(?:the\s+)?target", re.IGNORECASE,
)
FORECAST_SIGNAL = re.compile(
    r"\bforecast\b|outlook|guidance|expect(?:ed|s|ation)?|anticipate|project(?:ed|ion)?",
    re.IGNORECASE,
)
DEFINITION_SIGNAL = re.compile(r"\bdefinition\b|defined as|means|refers to", re.IGNORECASE)
SCOPE_CUSTOMER_SIGNAL = re.compile(r"\bcustomer\b|client", re.IGNORECASE)
SCOPE_SEGMENT_SIGNAL = re.compile(
    r"\bsegment\b|division|business unit|product line|product-specific", re.IGNORECASE,
)
SCOPE_GEOGRAPHY_SIGNAL = re.compile(
    r"\bgeograph|region|country|emea|asia|europe|united states|china|japan", re.IGNORECASE,
)
SCOPE_PRODUCT_SIGNAL = re.compile(r"\bproduct\b|system|unit model", re.IGNORECASE)
TOTAL_COMPANY_SIGNAL = re.compile(r"\btotal\b|company[- ]wide|group", re.IGNORECASE)
COUNT_METRIC_SIGNAL = re.compile(
    r"\b(?:count|number|units?|systems?|employees?|fte?s?|headcount|sold|shipped)\b",
    re.IGNORECASE,
)
RATE_UNIT_SIGNAL = re.compile(
    r"%|per[-\s]?hour|per\s+\w+|rate|ratio|intensity|throughput|efficiency", re.IGNORECASE,
)

FTE_COMPANY_WIDE = re.compile(
    r"total\s+employees?|total\s+workforce|all\s+employees?|"
    r"total\s+number\s+of\s+(?:payroll\s+)?employees|internal\s+employees?",
    re.IGNORECASE,
)
FTE_AVG_PAYROLL = re.compile(
    r"\baverage\b|\bpayroll\b|\btemporary\b|\bpermanent\b",
    re.IGNORECASE,
)
FTE_HEADCOUNT = re.compile(r"\bheadcount\b", re.IGNORECASE)
FTE_SPECIFIC = re.compile(
    r"dedicated\s+fte|team\s+fte|program\s+fte|project\s+fte",
    re.IGNORECASE,
)
FTE_SIGNAL = re.compile(
    r"\bfte?s?\b|full[- ]time\s+equivalents?|headcount|employees?|workforce|"
    r"payroll|permanent|temporary|internal|external|contractors?|turnover|attrition",
    re.IGNORECASE,
)
FTE_NON_EMPLOYEE = re.compile(
    r"survey\s+score|engagement\s+score|training\s+hours?|lost\s+time|incident\s+rate|"
    r"revenue|net\s+sales|dividend|buyback|emissions?|scope\s+[123]|net[- ]zero",
    re.IGNORECASE,
)
SUST_SIGNAL = re.compile(
    r"emissions?|greenhouse\s+gas|\bghg\b|co2e?|co₂e?|scope\s+[123]|net[- ]zero|"
    r"carbon|methane|flaring|renewable|energy\s+(?:efficien|savings?|use)|"
    r"electricity|power\s+consumption|wafer|waste|reuse|re-?use|recycl|circular|water|"
    r"biodiversity|supplier\s+sustainability|"
    r"diversity|inclusion|safety|ethics|governance",
    re.IGNORECASE,
)
SUST_TARGET_SIGNAL = re.compile(
    r"target|goal|ambition|commitment|committed|aim|reduce|reduction|halve|"
    r"achieve|become|maintain|eliminate|by\s+20[2-9]\d|20[3-9]\d",
    re.IGNORECASE,
)
SUST_BUSINESS_ONLY = re.compile(
    r"lng\s+sales|liquids?\s+production|barrels?\s+per\s+day|refining\s+throughput|"
    r"production\s+growth|volume\s+growth|market\s+share|revenue\s+growth|"
    r"customer\s+growth|stores?|branches?|locations?",
    re.IGNORECASE,
)
ESG_SIGNAL = re.compile(
    r"emissions?|greenhouse\s+gas|\bghg\b|co2e?|co₂e?|scope\s+[123]|renewable|"
    r"energy|waste|reuse|re-?use|recycl|water|circular|biodiversity|supplier|diversity|"
    r"inclusion|safety|ethics|governance",
    re.IGNORECASE,
)
FIN_SIGNAL = re.compile(
    r"net\s+sales|revenue|total\s+income|gross\s+profit|gross\s+margin|"
    r"operating\s+(?:income|profit)|income\s+from\s+operations|\bebit(?:da)?\b|"
    r"net\s+(?:income|profit)|income\s+tax|effective\s+tax\s+rate|\betr\b|"
    r"earnings\s+per\s+share|\beps\b|r&d|research\s+and\s+development|"
    r"free\s+cash\s+flow|operating\s+cash\s+flow|cash\s+flow\s+from\s+operat|"
    r"net\s+cash\s+provided\s+by\s+operating\s+activities|"
    r"cash\s+and\s+cash\s+equivalents|short[- ]term\s+investments?|"
    r"capex|capital\s+expenditure|property,\s+plant\s+and\s+equipment|"
    r"intangible\s+assets|return\s+on\s+(?:equity|invested\s+capital)|"
    r"\broe\b|\broic\b|\bcet1\b|capital\s+ratio|liquidity\s+coverage|"
    r"net\s+interest\s+margin|\bnim\b",
    re.IGNORECASE,
)
BIZ_SIGNAL = re.compile(
    r"systems?\s+sold|systems?\s+recognized|lithography\s+systems?|euv\s+systems?|"
    r"units?\s+sold|installed\s+base|"
    r"order\s+intake|order\s+book|backlog|bookings|customers?|clients?|"
    r"customer\s+satisfaction|suppliers?|reuse\s+rate|market\s+share|"
    r"production\s+volume|deliveries|loans?|deposits?|mortgages?|lng|"
    r"barrels?\s+per\s+day|refining\s+throughput|beer\s+volume|hectoliters?|"
    r"stores?|branches?|locations?|assets\s+under\s+management|\baum\b|"
    r"transaction\s+volume",
    re.IGNORECASE,
)
SH_SIGNAL = re.compile(
    r"dividend|share\s+buybacks?|share\s+repurchases?|repurchased|"
    r"returned?\s+to\s+shareholders?|shareholder\s+(?:returns?|distributions?)|"
    r"capital\s+return|payout\s+ratio|treasury\s+shares?|shares?\s+cancelled|"
    r"cash\s+returned",
    re.IGNORECASE,
)
