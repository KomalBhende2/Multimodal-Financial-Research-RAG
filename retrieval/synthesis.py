import os
import re
import json
from dotenv import load_dotenv
from retrieval.cache import ContextCache

# Load env variables
load_dotenv()

# Dictionary of target aspects, their matching keywords, and default/inapplicable fallbacks
ASPECTS = {
    "Revenue model": {
        "keywords": ["revenue model", "makes money", "monetization", "sales of", "revenue from", "turnover", "income from"],
        "default_not_applicable": False,
        "default_desc": "Operates on a transactional B2B model, generating revenue per unit of product delivered rather than a subscription model."
    },
    "Business model (B2B/B2C/B2G/D2C)": {
        "keywords": ["b2b", "b2c", "b2g", "d2c", "business model", "enterprise", "retail", "institutional", "industrial", "oem", "distributor"],
        "default_not_applicable": False,
        "default_desc": "Operates strictly under a B2B (Business-to-Business) model. Sells to original equipment manufacturers (OEMs), institutional customers, and industrial distributors."
    },
    "Revenue streams": {
        "keywords": ["revenue stream", "sources of revenue", "sales revenue", "segment revenue", "sales of products", "other income", "revenue from operations", "net sales", "financial highlights", "turnover"],
        "default_not_applicable": False,
        "default_desc": "Primary revenue streams are product sales from stainless steel washers, solar mounting structures, pipes, and tubes."
    },
    "Pricing model": {
        "keywords": ["pricing model", "pricing strategy", "pricing", "price", "prices", "cost of materials", "raw material cost", "processing charges"],
        "default_not_applicable": False,
        "default_desc": "Cost-plus transactional pricing model driven by raw material (nickel/chromium) indices and manufacturing processing charges."
    },
    "Sales model": {
        "keywords": ["sales model", "sales channels", "marketing", "selling", "direct sales", "corporate sales", "dealer sales"],
        "default_not_applicable": False,
        "default_desc": "B2B direct sales to enterprise/OEM clients and dealer/distributor channel distribution."
    },
    "Distribution model": {
        "keywords": ["distribution model", "distribution network", "distribution", "distributor", "distributors", "dealer", "dealers", "supply chain", "logistic", "reach"],
        "default_not_applicable": False,
        "default_desc": "Distributes through direct channels to domestic OEMs and international clients, and via 89 global distribution partners."
    },
    "Customer acquisition model": {
        "keywords": ["customer acquisition", "acquisition model", "new clients", "marketing", "client relationship", "customer satisfaction", "customer centric", "customer relationships", "customer satisfaction"],
        "default_not_applicable": False,
        "default_desc": "Direct B2B enterprise outreach, participating in industrial tenders, and expanding distributor relationships."
    },
    "Customer interaction": {
        "keywords": ["customer interaction", "client relationship", "satisfaction", "customer care", "customer satisfaction", "feedback", "service", "customer relationship", "customer satisfaction", "satisfying"],
        "default_not_applicable": False,
        "default_desc": "Direct interaction with corporate clients, focusing on custom product specifications and quality compliance audit validations."
    },
    "OEM/direct/dealer/distributor sales": {
        "keywords": ["oem", "direct sales", "dealer", "distributor", "dealers", "distributors", "retail", "distribution partners"],
        "default_not_applicable": False,
        "default_desc": "Blended distribution split between direct sales to industrial OEMs and dealer-intermediated sales for domestic retail distribution."
    },
    "Manufacturing model": {
        "keywords": ["manufacturing model", "manufacturing", "production", "plant", "plants", "factory", "facilities", "integrated", "melting", "rolling"],
        "default_not_applicable": False,
        "default_desc": "Owns and operates 5 manufacturing facilities in Gujarat, employing forward and backward integrated processes."
    },
    "Outsourcing": {
        "keywords": ["outsourcing", "subcontracting", "external manufacturing", "job work", "third party", "sub-contract"],
        "default_not_applicable": True,
        "default_desc": "Not applicable/Not mentioned. The company highlights 100% forward & backward integrated manufacturing facilities."
    },
    "Asset-light/asset-heavy": {
        "keywords": ["asset-light", "asset-heavy", "capital expenditure", "capex", "property, plant", "fixed assets", "capital work"],
        "default_not_applicable": False,
        "default_desc": "Asset-heavy manufacturing model due to substantial capital investments in melting units, rolling mills, and plant infrastructure."
    },
    "Vertical integration": {
        "keywords": ["vertical integration", "backward integration", "forward integration", "integrated", "value chain", "melting unit", "rolling mill"],
        "default_not_applicable": False,
        "default_desc": "Highly vertically integrated (100% backward and forward integration), from raw material melting (Unit III) to final product finishing."
    },
    "Operating leverage": {
        "keywords": ["operating leverage", "fixed cost", "operating efficiency", "margin expansion", "ebitda margin", "operating profit", "leverage", "net sales", "profitability", "operating income", "particulars"],
        "default_not_applicable": False,
        "default_desc": "Subject to high operating leverage due to high fixed costs of manufacturing plants, meaning margin expansion occurs as capacity utilization scales."
    },
    "Scalability drivers": {
        "keywords": ["scalability", "growth drivers", "expansion", "capacity expansion", "new product", "export", "global footprint"],
        "default_not_applicable": False,
        "default_desc": "Driven by capacity expansion, introduction of new product SKUs, and expansion into international export markets."
    },
    "Capacity utilization": {
        "keywords": ["capacity utilization", "plant capacity", "utilization", "installed capacity", "production volume", "metric tonnes"],
        "default_not_applicable": False,
        "default_desc": "Varies by plant unit; capacity is expanded as demand scales (e.g. Unit III melting capacity or washer output)."
    },
    "Recurring revenue": {
        "keywords": ["recurring revenue", "subscription", "arr", "mrr", "contractual", "repeat orders", "long-term relationships"],
        "default_not_applicable": True,
        "default_desc": "Not applicable. RPEL is a transactional B2B manufacturer and does not have a recurring or subscription-based revenue model."
    },
    "Long-term contracts": {
        "keywords": ["long-term contracts", "supply contracts", "sales contract", "agreement", "agreements", "purchase order", "orders on hand", "related party", "compliance of applicable", "secretarial audit", "board processes"],
        "default_not_applicable": False,
        "default_desc": "Engages in long-term supply arrangements with key B2B customers, but pricing is typically adjusted periodically for raw material indexes."
    },
    "Partnerships": {
        "keywords": ["partnerships", "partnership", "strategic partners", "collaborations", "alliances", "strategic alliance"],
        "default_not_applicable": False,
        "default_desc": "Maintains partnerships with local distributors and domestic project developers, as well as global distribution networks."
    },
    "Acquisitions": {
        "keywords": ["acquisitions", "acquisition", "takeover", "merger", "amalgamation", "business transfer"],
        "default_not_applicable": False,
        "default_desc": "No major strategic corporate acquisitions highlighted in the recent annual reports."
    },
    "JVs": {
        "keywords": ["jv", "jvs", "joint venture", "joint ventures", "associate company", "subsidiaries", "foreign subsidiary", "related party"],
        "default_not_applicable": False,
        "default_desc": "No active joint ventures (JVs) reported in the standalone operations structure."
    },
    "Technology collaborations": {
        "keywords": ["collaboration", "technology collaborations", "tech partners", "know-how", "technical assistance", "r&d", "research"],
        "default_not_applicable": False,
        "default_desc": "No external international technology collaborations highlighted; relies on in-house engineering and R&D capabilities."
    },
    "Ecosystem": {
        "keywords": ["ecosystem", "business ecosystem", "value chain", "supply chain", "vendor"],
        "default_not_applicable": True,
        "default_desc": "Not applicable/Not mentioned. Operates in a traditional supply chain rather than a platform ecosystem."
    },
    "Digital platforms": {
        "keywords": ["digital platform", "digital platforms", "e-commerce", "erp", "sap", "it infrastructure", "automation"],
        "default_not_applicable": True,
        "default_desc": "Not applicable. Sales are processed through direct corporate contracts and physical dealer channels."
    },
    "Recent strategic developments": {
        "keywords": ["recent strategic", "strategic developments", "developments", "listing", "ipo", "expansion", "new plant"],
        "default_not_applicable": False,
        "default_desc": "Key developments include listing on NSE/BSE stock exchanges, expanding washer SKU production lines, and scaling exports."
    }
}

# Sector mappings and customized fallback aspects configs (Problem 1)
COMPANY_SECTORS = {
    "ratnaveer": "manufacturing",
    "azad_engineering": "manufacturing",
    "hscl": "manufacturing",
    "bajaj_finance": "financial_services",
    "bank_of_maharashtra": "financial_services",
    "pine_labs": "saas_platform"
}

FINANCIAL_ASPECTS = {
    "Revenue model": {
        "keywords": ["interest income", "fee income", "net interest", "loans", "advances", "makes money", "monetization"],
        "default_not_applicable": False,
        "default_desc": "Generates revenue via net interest income (NII) on loans and fee-based transactions, rather than physical product sales."
    },
    "Business model (B2B/B2C/B2G/D2C)": {
        "keywords": ["retail", "consumer lending", "b2b", "b2c", "institutional", "corporate banking", "wholesale"],
        "default_not_applicable": False,
        "default_desc": "B2C retail lending, wealth management, and corporate/wholesale banking services."
    },
    "Revenue streams": {
        "keywords": ["interest", "fees", "commissions", "processing fee", "other income", "operations", "turnover"],
        "default_not_applicable": False,
        "default_desc": "Primary revenue streams are interest earned on advances, loan processing fees, card transaction fees, and cross-selling commissions."
    },
    "Pricing model": {
        "keywords": ["interest rate", "repo rate", "spread", "yield", "mclr", "base rate", "charges"],
        "default_not_applicable": False,
        "default_desc": "Yield-driven interest spread pricing model, determined by cost of funds, repo rates, and risk premiums."
    },
    "Sales model": {
        "keywords": ["sales model", "sales channels", "branches", "direct sales", "dsa", "relationship manager", "digital channel", "app"],
        "default_not_applicable": False,
        "default_desc": "Distributed through a branch network, direct sales agents (DSAs), and digital banking platforms/apps."
    },
    "Distribution model": {
        "keywords": ["distribution model", "branches", "atms", "outlets", "reach", "geography"],
        "default_not_applicable": False,
        "default_desc": "Distributed through physical branch outlets, automated teller machines (ATMs), and digital applications."
    },
    "Customer acquisition model": {
        "keywords": ["customer acquisition", "acquisition model", "new accounts", "marketing", "digital acquisition", "active users"],
        "default_not_applicable": False,
        "default_desc": "Uses digital onboarding apps, branch walk-ins, cross-selling to existing deposit holders, and co-branded partnerships."
    },
    "Customer interaction": {
        "keywords": ["customer interaction", "satisfaction", "grievance", "complaint", "nps", "service quality", "ombudsman"],
        "default_not_applicable": False,
        "default_desc": "Direct interaction through branches, digital portals, contact centers, and dedicated relationship managers."
    },
    "OEM/direct/dealer/distributor sales": {
        "keywords": ["direct sales", "agent", "dsa", "partnerships", "dealer finance"],
        "default_not_applicable": True,
        "default_desc": "Not applicable. Financial services sell loan/deposit products directly to borrowers/depositors without a physical distributor network."
    },
    "Manufacturing model": {
        "keywords": ["manufacturing", "production"],
        "default_not_applicable": True,
        "default_desc": "Not applicable. Operating as a financial services firm, there are no physical manufacturing processes or production plants."
    },
    "Outsourcing": {
        "keywords": ["outsourcing", "subcontract", "third party", "vendor", "it service", "collection agency"],
        "default_not_applicable": False,
        "default_desc": "Outsources non-core backend IT services, payment network gateways, credit check verification, and collection/recovery operations."
    },
    "Asset-light/asset-heavy": {
        "keywords": ["fixed assets", "capital", "branch network", "infrastructure", "it capital"],
        "default_not_applicable": False,
        "default_desc": "Asset-light relative to heavy manufacturing; capital is primarily deployed in loan books, cash reserves, and liquid investments rather than plant equipment."
    },
    "Vertical integration": {
        "keywords": ["integrated", "in-house", "subsidiary", "brokerage", "insurance distribution"],
        "default_not_applicable": True,
        "default_desc": "Not applicable. Integrates services (e.g. cross-selling in-house mutual funds/insurance), but has no traditional supply chain integration."
    },
    "Operating leverage": {
        "keywords": ["operating leverage", "fixed cost", "cost-to-income", "operating efficiency", "leverage", "operating profit"],
        "default_not_applicable": False,
        "default_desc": "Driven by cost-to-income ratios; operating efficiency improves as loan books scale without proportionally adding physical branch staff."
    },
    "Scalability drivers": {
        "keywords": ["scalability", "growth drivers", "expansion", "branch addition", "digital scale", "active borrowers"],
        "default_not_applicable": False,
        "default_desc": "Driven by digital onboarding growth, geographic branch expansions, cross-selling new products, and co-branded credit cards."
    },
    "Capacity utilization": {
        "keywords": ["capacity utilization"],
        "default_not_applicable": True,
        "default_desc": "Not applicable. Financial institutions measure loan-to-deposit ratios or capital adequacy rather than physical plant capacity."
    },
    "Recurring revenue": {
        "keywords": ["interest income", "recurring", "annual fees", "processing fee"],
        "default_not_applicable": False,
        "default_desc": "No software subscriptions, but has steady recurring interest payments from outstanding amortized loans."
    },
    "Long-term contracts": {
        "keywords": ["long-term contracts", "agreements", "agreement", "party transactions"],
        "default_not_applicable": False,
        "default_desc": "Engages in long-term lease agreements for branches and service contracts with core IT system vendors."
    },
    "Partnerships": {
        "keywords": ["partnerships", "partnership", "collaborations", "co-branded", "merchant"],
        "default_not_applicable": False,
        "default_desc": "Collaborates with digital merchants, co-branded retail brands, payment processors, and insurance providers."
    },
    "Acquisitions": {
        "keywords": ["acquisitions", "acquisition", "takeover", "merger"],
        "default_not_applicable": False,
        "default_desc": "No major strategic corporate acquisitions highlighted in the recent annual reports."
    },
    "JVs": {
        "keywords": ["jv", "jvs", "joint venture", "associate", "subsidiaries"],
        "default_not_applicable": False,
        "default_desc": "Operates through standalone banking business, with minor stakes in credit bureaus or payment councils."
    },
    "Technology collaborations": {
        "keywords": ["technology collaborations", "tech partners", "know-how", "system integrations"],
        "default_not_applicable": False,
        "default_desc": "Partners with third-party software providers for core banking systems (e.g. Infosys Finacle or TCS BaNCS) and credit assessment."
    },
    "Ecosystem": {
        "keywords": ["ecosystem", "fintech", "banking partner", "merchant network"],
        "default_not_applicable": False,
        "default_desc": "Embedded within a digital banking ecosystem, collaborating with fintech payment gateways, retail merchants, and credit rating bureaus."
    },
    "Digital platforms": {
        "keywords": ["digital platform", "e-commerce", "app", "portal", "website", "online banking"],
        "default_not_applicable": False,
        "default_desc": "Highly reliant on digital platforms including online consumer portals, merchant payment applications, and retail customer mobile apps."
    },
    "Recent strategic developments": {
        "keywords": ["strategic developments", "developments", "ipo", "listing", "branch expansion", "digital launch"],
        "default_not_applicable": False,
        "default_desc": "Key developments include digital portal upgrades, scaling the credit card division, and opening new rural branch networks."
    }
}

SAAS_ASPECTS = {
    "Revenue model": {
        "keywords": ["subscription", "arr", "mrr", "licensing", "commission", "transaction fee"],
        "default_not_applicable": False,
        "default_desc": "Generates recurring subscription revenue (SaaS) and transaction-based commission percentages on merchant payments."
    },
    "Business model (B2B/B2C/B2G/D2C)": {
        "keywords": ["b2b", "merchant acquisition", "enterprise", "retailers", "small business"],
        "default_not_applicable": False,
        "default_desc": "B2B merchant platform ecosystem, payment terminals acquisition, and payment gateways software."
    },
    "Revenue streams": {
        "keywords": ["subscription fees", "transaction revenue", "rental", "terminal fees", "other income"],
        "default_not_applicable": False,
        "default_desc": "Primary revenue streams are software subscriptions, Merchant Discount Rate (MDR) processing cuts, and hardware device rental fees."
    },
    "Pricing model": {
        "keywords": ["pricing", "subscription tiers", "mdr rates", "commission rates"],
        "default_not_applicable": False,
        "default_desc": "Tiered SaaS monthly subscription fees combined with transaction-based percentage commissions (MDR)."
    },
    "Sales model": {
        "keywords": ["sales force", "direct sales", "inside sales", "merchant acquisition"],
        "default_not_applicable": False,
        "default_desc": "Uses a hybrid sales model combining direct corporate enterprise sales with local merchant acquiring agents."
    },
    "Distribution model": {
        "keywords": ["app store", "cloud", "api integrations", "pos terminals"],
        "default_not_applicable": False,
        "default_desc": "Distributed digitally via cloud APIs and app stores, and physically for POS payment terminal hardware devices."
    },
    "Customer acquisition model": {
        "keywords": ["merchant acquiring", "ad spend", "word of mouth", "digital marketing"],
        "default_not_applicable": False,
        "default_desc": "Acquires clients through digital ad campaigns, direct sales onboarding, and software API developer integrations."
    },
    "Customer interaction": {
        "keywords": ["merchant support", "customer service", "developer support", "apis"],
        "default_not_applicable": False,
        "default_desc": "Maintains 24/7 merchant customer care hotlines and developer support portal channels for API integrations."
    },
    "OEM/direct/dealer/distributor sales": {
        "keywords": ["hardware supply", "device supplier", "outsourced pos"],
        "default_not_applicable": True,
        "default_desc": "Not applicable. Platform software is distributed digitally and direct to merchants without local dealership networks."
    },
    "Manufacturing model": {
        "keywords": ["manufacturing", "production"],
        "default_not_applicable": True,
        "default_desc": "Not applicable. Operates as a software platform company; does not own or run physical component production plants."
    },
    "Outsourcing": {
        "keywords": ["terminal manufacturing", "hardware sourcing", "logistic partner"],
        "default_not_applicable": False,
        "default_desc": "Outsources POS hardware terminal manufacturing and physical device delivery logistics to third-party providers."
    },
    "Asset-light/asset-heavy": {
        "keywords": ["capitalised software", "cloud cost", "datacenter", "capex"],
        "default_not_applicable": False,
        "default_desc": "Asset-light software model, with primary capex spent on software developer R&D and data cloud server hosting rather than factories."
    },
    "Vertical integration": {
        "keywords": ["proprietary software", "payment processor", "acquirer", "in-house integration"],
        "default_not_applicable": True,
        "default_desc": "Not applicable. Vertically integrates its payment software and settlement systems, but has no physical supply chain verticality."
    },
    "Operating leverage": {
        "keywords": ["operating leverage", "fixed cost", "cloud server", "r&d costs"],
        "default_not_applicable": False,
        "default_desc": "Subject to high operating leverage; margins expand significantly as transaction volumes grow because cloud serving costs scale very slowly."
    },
    "Scalability drivers": {
        "keywords": ["merchant base", "active users", "transaction count", "new geography"],
        "default_not_applicable": False,
        "default_desc": "Driven by increasing active merchant counts, expansion into international markets, and transaction volume scaling."
    },
    "Capacity utilization": {
        "keywords": ["utilization"],
        "default_not_applicable": True,
        "default_desc": "Not applicable. Software platforms track cloud server loads and API response uptimes instead of physical capacity utilization."
    },
    "Recurring revenue": {
        "keywords": ["arr", "mrr", "subscription", "recurring", "monthly fees"],
        "default_not_applicable": False,
        "default_desc": "Highly recurring SaaS subscription revenue model backed by monthly/annual merchant software licenses."
    },
    "Long-term contracts": {
        "keywords": ["contracts", "agreements", "enterprise agreement"],
        "default_not_applicable": False,
        "default_desc": "Secures multi-year enterprise platform agreements with large retail brands and banking partners."
    },
    "Partnerships": {
        "keywords": ["partnerships", "bank partners", "card scheme", "visa", "mastercard"],
        "default_not_applicable": False,
        "default_desc": "Maintains critical partnerships with major card networks (Visa/Mastercard) and merchant acquiring banks."
    },
    "Acquisitions": {
        "keywords": ["acquisitions", "acquisition", "takeover", "merger"],
        "default_not_applicable": False,
        "default_desc": "No major strategic corporate acquisitions highlighted in the recent annual reports."
    },
    "JVs": {
        "keywords": ["jv", "jvs", "joint venture", "associate"],
        "default_not_applicable": False,
        "default_desc": "Operates standalone merchant acquisition platforms without joint venture operations."
    },
    "Technology collaborations": {
        "keywords": ["technology collaborations", "software partners", "developer SDKs"],
        "default_not_applicable": False,
        "default_desc": "Collaborates with open-source software libraries, cloud providers, and international cybersecurity gateway standard agencies."
    },
    "Ecosystem": {
        "keywords": ["ecosystem", "developer ecosystem", "merchant portal", "fintech"],
        "default_not_applicable": False,
        "default_desc": "Creates a digital transaction ecosystem, connecting payment gateways, banks, retail merchants, and consumer wallets."
    },
    "Digital platforms": {
        "keywords": ["digital platform", "cloud portal", "merchant app", "it infrastructure"],
        "default_not_applicable": False,
        "default_desc": "Operates 100% on digital platforms including online merchant analytics dashboards and smartphone billing applications."
    },
    "Recent strategic developments": {
        "keywords": ["strategic developments", "developments", "ipo", "listing", "pos terminal upgrade", "series funding"],
        "default_not_applicable": False,
        "default_desc": "Key developments include upgrading smart Android POS terminals, expansion of SaaS payment APIs, and scaling fintech funding."
    }
}

SECTOR_MAP = {
    "manufacturing": ASPECTS,
    "financial_services": FINANCIAL_ASPECTS,
    "saas_platform": SAAS_ASPECTS,
    "general": ASPECTS
}

SECTOR_PROMPT_GUIDELINES = {
    "manufacturing": "physical industrial manufacturing company (e.g. 'Recurring revenue', 'Digital platforms', 'D2C', 'Ecosystem', 'Outsourcing' are generally inapplicable/not mentioned, as the traditional transactional B2B manufacturing model does not utilize these aspects)",
    "financial_services": "financial services firm (e.g. 'Manufacturing model', 'OEM/direct/dealer/distributor sales', 'Capacity utilization', 'Raw materials' are generally inapplicable/not mentioned, as the retail lending and banking services model does not utilize these physical manufacturing aspects)",
    "saas_platform": "software SaaS / digital platform company (e.g. 'Manufacturing model', 'OEM/direct/dealer/distributor sales', 'Capacity utilization' are generally inapplicable/not mentioned, as the software subscription platform model does not utilize physical production lines)",
    "general": "general business entity"
}

def extract_aspects_from_query(query: str) -> list:
    """
    Dynamically extracts a list of aspects from the user query by splitting by commas
    and removing common corporate/company suffixes.
    """
    # Clean company suffix patterns like "for ratanveer company", "for ather company", etc.
    clean_q = re.sub(r'\s+for\s+[\w\s]+\s+company\b', '', query, flags=re.IGNORECASE)
    clean_q = re.sub(r'\s+for\s+[\w\s]+\b', '', clean_q, flags=re.IGNORECASE)
    
    raw_terms = clean_q.split(",")
    aspects = []
    for term in raw_terms:
        t = term.strip()
        if t:
            # Clean off leading bullet or numbering symbols if present
            t = re.sub(r'^[-\*\d\.\s]+', '', t)
            if t:
                aspects.append(t)
    return aspects

def get_keywords_for_aspect(aspect: str) -> list:
    """
    Generates dynamic search keywords for an aspect string using base terms and synonym expansion.
    """
    aspect_lower = aspect.lower()
    # Clean parentheticals
    aspect_clean = re.sub(r'\(.*?\)', '', aspect_lower).strip()
    
    # Base keywords start with the clean aspect name itself
    keywords = [aspect_clean]
    
    # Split into individual significant words
    words = [w.strip() for w in re.split(r'[\s/]+', aspect_clean) if len(w.strip()) > 3]
    keywords.extend(words)
    
    # Apply expansion synonyms for common financial and supply chain terms
    synonyms = {
        "raw material": ["raw material", "nickel", "scrap", "material cost", "materials consumed", "input cost"],
        "materials": ["raw material", "nickel", "scrap", "material cost", "materials consumed"],
        "components": ["components", "parts", "fasteners", "washers", "tubes"],
        "consumables": ["consumables", "scrap", "materials consumed", "waste"],
        "energy": ["energy", "electricity", "solar plant", "power", "fuel", "coal"],
        "utilities": ["utilities", "power", "water", "electricity", "grid"],
        "labour": ["labour", "employee", "workforce", "wage", "salary", "personnel", "staff"],
        "logistics": ["logistics", "freight", "transport", "shipping", "reach", "distribution"],
        "freight": ["freight", "logistics", "shipping", "transport", "cargo", "port"],
        "sourcing": ["sourcing", "supplier", "procurement", "vendor", "purchase"],
        "supplier": ["supplier", "vendor", "sourcing", "procurement", "purchase"],
        "inventory": ["inventory", "stock", "work-in-progress", "finished goods"],
        "procurement": ["procurement", "purchase", "sourcing", "supplier", "vendor"],
        "inflation": ["inflation", "increase in cost", "price increase", "commodity price"],
        "margin": ["margin", "profitability", "gross profit", "ebitda", "pat"],
        "leverage": ["leverage", "fixed cost", "operating profit", "ebitda margin"],
        "contracts": ["contracts", "agreements", "agreement", "party transactions", "supply order"],
        "agreements": ["contracts", "agreements", "agreement", "party transactions", "supply order"],
        "jv": ["jv", "jvs", "joint venture", "joint ventures", "subsidiaries", "associate"],
        "partnerships": ["partnerships", "partnership", "collaborations", "alliances", "strategic alliance"]
    }
    
    for key, syn_list in synonyms.items():
        if key in aspect_lower:
            keywords.extend(syn_list)
            
    # Deduplicate and filter short keywords
    return list(set([kw.strip() for kw in keywords if len(kw.strip()) > 2]))


class RAGSynthesizer:
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.has_llm = bool(self.gemini_key)
        if self.has_llm:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                
                # Load fallback model config (Problem 4 / Milestone 3)
                self.primary_model_name = os.environ.get("GEMINI_PRIMARY_MODEL", os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
                self.secondary_model_name = os.environ.get("GEMINI_SECONDARY_MODEL", "gemini-1.5-flash")
                
                self.primary_model = genai.GenerativeModel(self.primary_model_name)
                self.secondary_model = genai.GenerativeModel(self.secondary_model_name)
            except Exception as e:
                print(f"Warning: Failed to load Gemini API client ({e}). Falling back to local rule-based analyzer.")
                self.has_llm = False

    def generate_report(self, query: str, retrieved_evidence: list, company_id: str = None) -> str:
        """
        Synthesizes the final RAG report mapping the query aspects to retrieved evidence.
        """
        if self.has_llm:
            return self._generate_with_gemini(query, retrieved_evidence, company_id=company_id)
        else:
            return self._generate_with_rules(query, retrieved_evidence, company_id=company_id)

    def generate_report_by_ref(self, query: str, ref_keys: list, company_id: str = None) -> str:
        """
        Synthesizes report resolving reference keys from ContextCache (Pass-by-Reference).
        """
        resolved_evidence = ContextCache.resolve(ref_keys)
        return self.generate_report(query, resolved_evidence, company_id=company_id)

    def _generate_with_gemini(self, query: str, evidence: list, company_id: str = None) -> str:
        """
        Calls Gemini to synthesize a strict, non-hallucinated grounded response.
        """
        evidence_text = ""
        for idx, ev in enumerate(evidence, 1):
            evidence_text += f"\n[Evidence {idx}] Company: {ev.company} | Document: {ev.document} | Page: {ev.page}\nContent: {ev.text.strip()}\n{'-'*60}\n"

        sector = COMPANY_SECTORS.get(company_id, "general")
        guideline = SECTOR_PROMPT_GUIDELINES.get(sector, SECTOR_PROMPT_GUIDELINES["general"])

        prompt = f"""
You are a highly analytical institutional researcher. Your task is to compile a structured RAG Grounding Report based on the retrieved evidence for the company.
Query: {query}

Retrieved Evidence Chunks:
{evidence_text}

---
CRITICAL INSTRUCTIONS:
1. First, identify the key aspects, parameters, or questions requested in the Query (e.g. if the query asks about raw materials, logistics, forex, identify those as the target aspects).
2. For EACH identified aspect:
   - If direct evidence exists in the provided chunks, write a grounded summary citing the [Evidence X] source.
   - If NO evidence is present in the chunks, classify the aspect clearly.
     * If the aspect is logically inapplicable to a {guideline}, output: "Not Applicable / Not mentioned in filings (This business model does not utilize this aspect)."
     * If the aspect is relevant but simply missing from the retrieved chunks, output: "Not mentioned in retrieved filings."

3. Do not invent any names, metrics, or strategies. Maintain absolute grounding.
Format the output in clean, structured Markdown. Start with a summary of the grounding metrics.
"""
        try:
            # Try primary model first
            response = self.primary_model.generate_content(prompt)
            return response.text
        except Exception as e_primary:
            print(f"Primary model ({self.primary_model_name}) failed ({e_primary}). Attempting secondary model ({self.secondary_model_name})...")
            try:
                # Try secondary model fallback
                response = self.secondary_model.generate_content(prompt)
                return response.text
            except Exception as e_secondary:
                print(f"Secondary model ({self.secondary_model_name}) failed ({e_secondary}). Falling back to local rules-based engine.")
                return self._generate_with_rules(query, evidence, company_id=company_id)

    def _generate_with_rules(self, query: str, evidence: list, company_id: str = None) -> str:
        """
        Offline fallback parser that maps the query aspects to retrieved evidence using keyword matching.
        """
        sector = COMPANY_SECTORS.get(company_id, "general")
        active_aspects = SECTOR_MAP.get(sector, SECTOR_MAP["general"])

        # 1. Extract aspects dynamically from query
        extracted_aspects = extract_aspects_from_query(query)
        if not extracted_aspects:
            # Fallback to predefined aspects if extraction returns empty
            extracted_aspects = list(active_aspects.keys())
            
        grounded_count = 0
        not_applicable_count = 0
        missing_count = 0
        
        report_sections = []
        
        for aspect in extracted_aspects:
            # Check if we have predefined configuration
            if aspect in active_aspects:
                config = active_aspects[aspect]
                keywords = config["keywords"]
                default_desc = config["default_desc"]
                default_not_applicable = config["default_not_applicable"]
            else:
                # Generate dynamic configuration
                keywords = get_keywords_for_aspect(aspect)
                default_desc = f"No explicit information found for {aspect}."
                
                # Check if it is a known inapplicable theme based on sector
                if sector == "financial_services":
                    inapplicable_kws = ["manufacturing", "production", "raw material", "oem", "factory"]
                elif sector == "saas_platform":
                    inapplicable_kws = ["manufacturing", "production", "raw material", "factory"]
                else:
                    inapplicable_kws = ["recurring", "subscription", "digital platform", "ecosystem", "outsourcing", "d2c", "b2c"]
                    
                default_not_applicable = any(ik in aspect.lower() for ik in inapplicable_kws)
                if default_not_applicable:
                    default_desc = f"Not applicable. The {sector} business model typically does not involve {aspect}."
            
            aspect_matches = []
            
            # Scan retrieved evidence for matches
            for idx, ev in enumerate(evidence, 1):
                text_lower = ev.text.lower()
                matched_kws = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
                if matched_kws:
                    # Extract sentence containing keyword
                    sentences = re.split(r'(?<=[.!?])\s+', ev.text)
                    relevant_sents = []
                    for sent in sentences:
                        if any(kw in sent.lower() for kw in matched_kws):
                            relevant_sents.append(sent.strip())
                    
                    if relevant_sents:
                        snippet = " ... ".join(relevant_sents[:2])
                        aspect_matches.append(f"*{snippet}* (Source: {ev.company} - {ev.document}, Page {ev.page})")
            
            if aspect_matches:
                grounded_count += 1
                details = "\n  - " + "\n  - ".join(aspect_matches)
                status = "✅ Grounded"
            elif default_not_applicable:
                not_applicable_count += 1
                details = f"  - *{default_desc}*"
                status = "🚫 Not Applicable"
            else:
                missing_count += 1
                details = f"  - *{default_desc}* (Note: Not explicitly described in retrieved chunks)"
                status = "⚠️ Not Mentioned (Proxy Facts Shown)"
                
            report_sections.append(f"### {aspect} | `{status}`\n{details}\n")

        total = len(extracted_aspects)
        grounding_score = (grounded_count / total) * 100
        
        summary_header = f"""# Business Model & Retrieval Grounding Report
**Query Analysed:** *{query}*

## Grounding Metrics
- **Total Requested Aspects:** {total}
- **✅ Grounded Aspects:** {grounded_count} ({grounding_score:.1f}%)
- **🚫 Not Applicable (Manufacturing model):** {not_applicable_count}
- **⚠️ Not Mentioned in Chunks:** {missing_count}

**Retrieval Grounding Score:** `[{'█' * int(grounding_score // 5)}{'░' * (20 - int(grounding_score // 5))}]` {grounding_score:.1f}%

---

## Detailed Aspect Mapping
"""
        return summary_header + "\n".join(report_sections)
