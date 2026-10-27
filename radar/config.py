"""
Shared configuration for the SEA Boom Radar project.

Every notebook imports from here so that sector definitions, search terms and
backtest labels live in exactly one place.

LABEL CONVENTIONS
-----------------
label = "boom"       SF-origin model that became a durable, venture-scale sector in SEA
                     (appears as a tracked growth vertical in Google/Temasek/Bain
                     e-Conomy SEA reports and produced multiple SEA unicorns/leaders).
label = "noise"      Wave that was heavily hyped in SF but did NOT become a durable
                     SEA sector (collapsed, shut down or never got traction).
                     hype_peak = year of peak SF hype, dated from public history
                     (funding/press peak), not from the search data.
label = "ambiguous"  Reasonable people disagree -> EXCLUDED from accuracy scoring.
label = "candidate"  Live 2026 question. This is what the radar is trying to answer.

Labels are human judgement calls made from public history. They are written
down here, with a one-line rationale, BEFORE any model is run, so the backtest
cannot quietly move the goalposts. Challenge them; that is the point.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
for p in (RAW, PROC, FIG):
    p.mkdir(parents=True, exist_ok=True)

USER_AGENT = "sea-boom-radar/0.1 (research project)"

START_YEAR, END_YEAR = 2007, 2026          # HN starts 2007; data runs to Sep 2026
GT_TIMEFRAME = "2004-01-01 2026-09-30"

# Google Trends geographies. US-CA-807 = San Francisco-Oakland-San Jose DMA.
SF_GEO = "US-CA-807"
US_GEO = "US"
SEA_GEOS = {"SG": "Singapore", "ID": "Indonesia", "VN": "Vietnam",
            "PH": "Philippines", "TH": "Thailand", "MY": "Malaysia"}
SEA_WIKIS = ["id", "vi", "th", "ms", "tl"]

# ---------------------------------------------------------------------------
# Sector definitions
#   gt_sf / gt_sea : Google Trends query ("+" = OR). SEA adds local brands/terms.
#   hn             : Hacker News phrases (SF builder/investor attention)
#   wiki           : English Wikipedia article (SEA-language versions found via Wikidata)
#   yc             : regex over YC tags + one-liner (founder supply)
#   gh             : GitHub repo search phrase (developer/builder activity)
#   oa             : OpenAlex title/abstract phrase (research activity)
#   sea_boom_year  : for labelled booms, year the sector broke out in SEA
# ---------------------------------------------------------------------------
SECTORS = {
    # ---------------- Labelled BOOMS (SF -> SEA transfer happened) ----------------
    "ride_hailing": dict(
        label="boom", sea_boom_year=2015,
        why="Uber/Lyft (SF 2010-12) -> Grab, Gojek became SEA super-apps; e-Conomy SEA 'transport' vertical.",
        gt_sf="uber + lyft + ride sharing", gt_sea="grab + gojek + ride hailing + uber",
        hn=["ridesharing", "ride sharing", "ride-hailing"], wiki="Ridesharing company",
        yc=r"ride.?shar|ride.?hail|rideshare|carpool|(?<!air )(?<!robo)\btaxi", gh="ridesharing", oa="ride-hailing"),
    "food_delivery": dict(
        label="boom", sea_boom_year=2018,
        why="DoorDash/Postmates (SF 2013-15) -> GrabFood, GoFood, foodpanda; e-Conomy SEA 'food delivery' vertical.",
        gt_sf="doordash + postmates + food delivery + grubhub", gt_sea="grabfood + gofood + foodpanda + food delivery",
        hn=["food delivery"], wiki="Food delivery",
        yc=r"food deliver|meal deliver|restaurant deliver", gh="food delivery", oa="food delivery platform"),
    "digital_payments": dict(
        label="boom", sea_boom_year=2017,
        why="Square/Venmo/Stripe (SF 2010-14) -> GoPay, OVO, GCash, MoMo; e-Conomy SEA 'digital payments'.",
        gt_sf="venmo + mobile payment + apple pay", gt_sea="e-wallet + gopay + gcash + dompet digital + ví điện tử",
        hn=["mobile payments", "mobile wallet", "digital wallet"], wiki="Mobile payment",
        yc=r"payment|wallet", gh="mobile payment", oa="mobile payment"),
    "bnpl": dict(
        label="boom", sea_boom_year=2020,
        why="Affirm (SF 2012, scaled 2018-20) -> Atome, Kredivo, Akulaku; SEA 'digital lending' growth.",
        gt_sf="buy now pay later + affirm + afterpay + klarna", gt_sea="paylater + buy now pay later + atome + kredivo + akulaku",
        hn=["buy now pay later", "BNPL"], wiki="Buy now, pay later",
        yc=r"buy now.?pay later|bnpl|point.of.sale financ", gh="buy now pay later", oa="buy now pay later"),
    "digital_banks": dict(
        label="boom", sea_boom_year=2021,
        why="Chime/neobanks (SF 2017-20) -> MAS digital bank licences 2020, GXS, Trust, Bank Jago, SeaBank.",
        gt_sf="neobank + chime bank + digital bank", gt_sea="digital bank + bank digital + seabank + gxs + bank jago",
        hn=["neobank", "challenger bank"], wiki="Neobank",
        yc=r"neobank|neo.?bank|challenger bank|digital bank", gh="neobank", oa="neobank"),
    "retail_investing": dict(
        label="boom", sea_boom_year=2021,
        why="Robinhood/robo-advisors (SF 2014-20) -> Ajaib, Bibit, Pluang, StashAway, Endowus.",
        gt_sf="robinhood app + robo advisor + betterment + wealthfront", gt_sea="stashaway + ajaib + bibit + investment app + robo advisor",
        hn=["robo-advisor", "robinhood app", "commission-free trading"], wiki="Robo-advisor",
        yc=r"robo.?advis|retail invest|stock trading|brokerage|wealth manag", gh="robo advisor", oa="robo-advisor"),
    "video_streaming": dict(
        label="boom", sea_boom_year=2016,
        why="Netflix streaming (2007-12) -> Netflix SEA launch 2016, iflix, Viu; e-Conomy SEA 'online media'.",
        gt_sf="netflix + hulu + streaming", gt_sea="netflix + iflix + viu + streaming",
        hn=["video streaming", "netflix streaming"], wiki="Streaming media",
        yc=r"video stream|streaming", gh="video streaming", oa="video streaming service"),

    # ---------------- Labelled NOISE (SF hype that did not transfer) ----------------
    "daily_deals": dict(
        label="noise", sea_boom_year=None, hype_peak=2011,
        why="Groupon/LivingSocial frenzy 2010-11; Groupon SEA exited, Ensogo shut 2016.",
        gt_sf="groupon + livingsocial + daily deals", gt_sea="groupon + ensogo + daily deals",
        hn=["groupon", "daily deals"], wiki="Group buying",
        yc=r"daily deal|group.?buy|coupon", gh="daily deals", oa="daily deals"),
    "coworking": dict(
        label="noise", sea_boom_year=None, hype_peak=2019,
        why="WeWork-as-tech thesis 2015-19; failed IPO 2019, SEA operators consolidated, no venture-scale outcome.",
        gt_sf="wework + coworking + coworking space", gt_sea="coworking + wework + justco + coworking space",
        hn=["coworking", "wework"], wiki="Coworking",
        yc=r"cowork|co-working|flexible office", gh="coworking", oa="coworking space"),
    "scooter_sharing": dict(
        label="noise", sea_boom_year=None, hype_peak=2018,
        why="Bird/Lime 2017-19; Singapore banned e-scooters on footpaths 2019, operators exited/pivoted.",
        gt_sf="bird scooter + lime scooter + scooter sharing", gt_sea="e-scooter + neuron scooter + beam scooter + scooter sharing",
        hn=["scooter sharing", "dockless scooters", "bird scooters"], wiki="Scooter-sharing system",
        yc=r"scooter shar|micromobility|micro.?mobility|dockless", gh="scooter sharing", oa="shared e-scooter"),
    "cloud_kitchens": dict(
        label="noise", sea_boom_year=None, hype_peak=2020,
        why="CloudKitchens/ghost kitchens 2019-21; SEA platforms scaled kitchen programmes back.",
        gt_sf="ghost kitchen + cloud kitchen + virtual restaurant", gt_sea="cloud kitchen + ghost kitchen + grabkitchen",
        hn=["ghost kitchen", "cloud kitchen", "dark kitchen"], wiki="Virtual restaurant",
        yc=r"ghost kitchen|cloud kitchen|virtual restaurant|dark kitchen", gh="ghost kitchen", oa="ghost kitchen"),
    "quick_commerce": dict(
        label="noise", sea_boom_year=None, hype_peak=2021,
        why="Gopuff/10-minute delivery 2020-22; Indonesia's Astro shut down 2023.",
        gt_sf="gopuff + quick commerce + 15 minute delivery", gt_sea="quick commerce + dark store + grabmart",
        hn=["quick commerce", "dark stores", "gopuff"], wiki="Q-commerce",
        yc=r"quick.?commerce|q-commerce|dark store|instant deliver|10.minute", gh="quick commerce", oa="quick commerce"),
    "nft": dict(
        label="noise", sea_boom_year=None, hype_peak=2021,
        why="NFT mania 2021; volumes collapsed >95% in 2022.",
        gt_sf="nft + nfts", gt_sea="nft + nfts",
        hn=["NFT", "NFTs"], wiki="Non-fungible token",
        yc=r"\bnft", gh="nft", oa="non-fungible token"),
    "metaverse": dict(
        label="noise", sea_boom_year=None, hype_peak=2022,
        why="Metaverse rebrand wave 2021-22; no durable SEA sector emerged.",
        gt_sf="metaverse", gt_sea="metaverse",
        hn=["metaverse"], wiki="Metaverse",
        yc=r"metaverse", gh="metaverse", oa="metaverse"),
    "play_to_earn": dict(
        label="noise", sea_boom_year=None, hype_peak=2021,
        why="Axie Infinity P2E 2021 (huge in PH/VN), collapsed after Ronin hack and token crash 2022.",
        gt_sf="play to earn + axie infinity + web3 gaming", gt_sea="play to earn + axie infinity + web3 gaming",
        hn=["play-to-earn", "axie infinity", "web3 gaming"], wiki="Blockchain game",
        yc=r"play.to.earn|web3 gam|blockchain gam|crypto gam", gh="play to earn", oa="play-to-earn"),
    "ibuying": dict(
        label="noise", sea_boom_year=None, hype_peak=2021,
        why="Opendoor/Zillow Offers 2018-21; Zillow exited 2021, model never reached SEA.",
        gt_sf="opendoor + ibuying + offerpad", gt_sea="opendoor + ibuying",
        hn=["opendoor", "ibuying", "ibuyer"], wiki="Opendoor",
        yc=r"ibuy|instant offer|home.?buying", gh="ibuying", oa="iBuying"),
    "social_audio": dict(
        label="noise", sea_boom_year=None, hype_peak=2021,
        why="Clubhouse 2021 surge, usage collapsed within ~12 months.",
        gt_sf="clubhouse app + twitter spaces + social audio", gt_sea="clubhouse app + twitter spaces + social audio",
        hn=["clubhouse app", "social audio", "twitter spaces"], wiki="Clubhouse (app)",
        yc=r"social audio|live audio|audio room", gh="social audio", oa="social audio"),
    "drone_delivery": dict(
        label="noise", sea_boom_year=None, hype_peak=2016,
        why="Prime Air hype from 2013; still not a venture-scale SEA sector by 2026.",
        gt_sf="drone delivery + amazon prime air", gt_sea="drone delivery",
        hn=["drone delivery"], wiki="Delivery drone",
        yc=r"drone deliver|delivery drone", gh="drone delivery", oa="drone delivery"),
    "consumer_3d_printing": dict(
        label="noise", sea_boom_year=None, hype_peak=2013,
        why="'3D printer in every home' wave 2012-14; consumer market stalled.",
        gt_sf="3d printer + 3d printing", gt_sea="3d printer + 3d printing",
        hn=["3D printing", "3D printer"], wiki="3D printing",
        yc=r"3d print", gh="3d printer", oa="3D printing"),

    # ---------------- AMBIGUOUS (excluded from accuracy metrics) ----------------
    "crypto_exchanges": dict(
        label="ambiguous", sea_boom_year=None,
        why="Large user bases (Indodax, Pintu, Coins.ph) but extreme boom-bust and regulatory churn.",
        gt_sf="coinbase + crypto exchange + buy bitcoin", gt_sea="indodax + pintu + tokocrypto + crypto exchange + binance",
        hn=["coinbase", "crypto exchange"], wiki="Cryptocurrency exchange",
        yc=r"crypto|bitcoin|blockchain", gh="crypto exchange", oa="cryptocurrency exchange"),
    "telemedicine": dict(
        label="ambiguous", sea_boom_year=None,
        why="Halodoc/Doctor Anywhere grew strongly in 2020, then cooled; jury still out.",
        gt_sf="telemedicine + telehealth", gt_sea="telemedicine + halodoc + alodokter + doctor anywhere + telehealth",
        hn=["telemedicine", "telehealth"], wiki="Telehealth",
        yc=r"telemedicine|telehealth|virtual care|online doctor", gh="telemedicine", oa="telemedicine"),
    "edtech": dict(
        label="ambiguous", sea_boom_year=None,
        why="Ruangguru scaled, but COVID edtech surge reversed (Zenius shut 2024).",
        gt_sf="online learning + edtech + coursera + online tutoring", gt_sea="ruangguru + edtech + online learning + belajar online",
        hn=["edtech", "online learning"], wiki="Educational technology",
        yc=r"edtech|education|tutor|learning platform", gh="online learning", oa="educational technology"),
    "insurtech": dict(
        label="ambiguous", sea_boom_year=None,
        why="Steady but sub-scale SEA outcomes (Qoala, PolicyStreet).",
        gt_sf="insurtech + lemonade insurance + online insurance", gt_sea="insurtech + qoala + online insurance + asuransi online",
        hn=["insurtech", "lemonade insurance"], wiki="Insurtech",
        yc=r"insurtech|insurance", gh="insurtech", oa="insurtech"),
    "p2p_lending": dict(
        label="ambiguous", sea_boom_year=None,
        why="Indonesian 'pinjol' boom 2018-20, then crackdowns and Investree collapse 2024.",
        gt_sf="lending club + peer to peer lending + prosper loans", gt_sea="pinjaman online + p2p lending + pinjol + funding societies",
        hn=["peer-to-peer lending", "lending club", "p2p lending"], wiki="Peer-to-peer lending",
        yc=r"peer.to.peer lend|p2p lend|marketplace lend", gh="p2p lending", oa="peer-to-peer lending"),
    "b2b_saas": dict(
        label="ambiguous", sea_boom_year=None,
        why="SEA SaaS exists but smaller/later than consumer verticals; definitional.",
        gt_sf="saas + software as a service", gt_sea="saas + software as a service",
        hn=["SaaS"], wiki="Software as a service",
        yc=r"\bsaas\b|b2b", gh="saas", oa="software as a service"),

    # ---------------- 2026 CANDIDATES (the actual question) ----------------
    "generative_ai": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="chatgpt + generative ai + midjourney", gt_sea="chatgpt + generative ai + ai generatif",
        hn=["generative AI", "ChatGPT", "GPT-3"], wiki="Generative AI",
        yc=r"generative ai|genai|\bllm|large language|gpt", gh="llm", oa="large language model"),
    "ai_agents": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="ai agent + ai agents + agentic ai", gt_sea="ai agent + ai agents + agentic ai",
        hn=["AI agents", "agentic", "autonomous agents"], wiki="AI agent",
        yc=r"ai agent|agentic|autonomous agent|ai employee|ai worker", gh="ai agent", oa="LLM agent"),
    "ai_coding": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="github copilot + cursor ai + vibe coding + ai coding", gt_sea="github copilot + cursor ai + vibe coding + ai coding",
        hn=["copilot", "AI coding", "vibe coding"], wiki="Vibe coding",
        yc=r"ai cod|code generation|coding assistant|developer tool", gh="coding assistant", oa="code generation"),
    "ai_datacenters": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="data center + gpu cloud + ai data center", gt_sea="data center + pusat data + trung tâm dữ liệu",
        hn=["data center", "GPU cloud"], wiki="Data center",
        yc=r"data cent|gpu cloud|gpu infra|compute infra", gh="gpu cluster", oa="data center energy"),
    "stablecoins": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="stablecoin + usdc + usdt", gt_sea="stablecoin + usdc + usdt + tether",
        hn=["stablecoin", "USDC"], wiki="Stablecoin",
        yc=r"stablecoin|usdc", gh="stablecoin", oa="stablecoin"),
    "embedded_finance": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="embedded finance + banking as a service", gt_sea="embedded finance + banking as a service",
        hn=["embedded finance", "banking-as-a-service", "banking as a service"], wiki="Banking as a service",
        yc=r"embedded finance|banking.as.a.service|baas|card issuing", gh="banking as a service", oa="embedded finance"),
    "carbon_markets": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="carbon credits + carbon offset + carbon market", gt_sea="carbon credits + carbon offset + kredit karbon",
        hn=["carbon credits", "carbon offsets", "carbon removal"], wiki="Carbon credit",
        yc=r"carbon credit|carbon offset|carbon remov|carbon account|decarboni", gh="carbon accounting", oa="carbon credit"),
    "electric_two_wheelers": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="electric motorcycle + e-bike + electric bike", gt_sea="motor listrik + xe máy điện + electric motorcycle + gogoro",
        hn=["electric motorcycle", "e-bike", "ebike"], wiki="Electric motorcycles and scooters",
        yc=r"electric motorcycle|e-bike|electric bike|electric scooter|battery swap", gh="electric motorcycle", oa="electric motorcycle"),
    "battery_storage": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="battery storage + home battery + powerwall", gt_sea="battery storage + solar battery + energy storage",
        hn=["battery storage", "grid storage", "energy storage"], wiki="Battery energy storage system",
        yc=r"battery stor|energy stor|grid stor", gh="battery management", oa="battery energy storage"),
    "cybersecurity": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="cybersecurity + cyber security", gt_sea="cybersecurity + cyber security + keamanan siber",
        hn=["cybersecurity"], wiki="Computer security",
        yc=r"cyber.?secur|security|threat detect", gh="cybersecurity", oa="cybersecurity"),
    "humanoid_robots": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="humanoid robot + tesla optimus + figure robot", gt_sea="humanoid robot + robot humanoid",
        hn=["humanoid robot", "humanoid robots", "Optimus robot"], wiki="Humanoid robot",
        yc=r"humanoid|general.purpose robot|robot foundation model", gh="humanoid robot", oa="humanoid robot"),
    "ai_healthcare": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="ai healthcare + ai doctor + medical ai + ai scribe", gt_sea="ai healthcare + ai doctor + medical ai",
        hn=["medical AI", "AI in healthcare", "clinical AI", "AI scribe"], wiki="Artificial intelligence in healthcare",
        yc=r"ai scribe|clinical ai|medical ai|ai for (?:health|clinic|doctor)|healthcare ai", gh="medical ai", oa="artificial intelligence healthcare"),
    "agritech": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="agtech + agritech + vertical farming + precision agriculture", gt_sea="agritech + efishery + pertanian digital + precision agriculture",
        hn=["agtech", "agritech", "vertical farming", "precision agriculture"], wiki="Agricultural technology",
        yc=r"agtech|agritech|agricultur|farm", gh="precision agriculture", oa="precision agriculture"),
    "creator_economy": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="creator economy + content creator + patreon", gt_sea="content creator + kreator konten + tiktok affiliate",
        hn=["creator economy", "patreon", "substack"], wiki="Creator economy",
        yc=r"creator|influencer", gh="creator economy", oa="creator economy"),
    "smb_vertical_saas": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="toast pos + square pos + pos system", gt_sea="pos system + moka pos + kasir online",
        hn=["vertical SaaS", "point of sale"], wiki="Vertical market software",
        yc=r"vertical saas|point.of.sale|\bpos\b|small business software|smb software", gh="point of sale", oa="point of sale system"),
    "social_commerce": dict(
        label="candidate", sea_boom_year=None, why="SEA-led trend (TikTok Shop, Shopee Live): tests reverse flow.",
        gt_sf="tiktok shop + live shopping + social commerce", gt_sea="tiktok shop + shopee live + live shopping + social commerce",
        hn=["live shopping", "social commerce", "tiktok shop"], wiki="Social commerce",
        yc=r"social commerce|live shopping|livestream shopping|live commerce", gh="social commerce", oa="social commerce"),
    "autonomous_vehicles": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="waymo + robotaxi + self driving car", gt_sea="robotaxi + self driving car + autonomous vehicle",
        hn=["robotaxi", "self-driving", "Waymo"], wiki="Self-driving car",
        yc=r"self.driving|autonomous vehicle|autonomous driving|robotaxi", gh="self driving car", oa="autonomous vehicle"),
    "prediction_markets": dict(
        label="candidate", sea_boom_year=None, why="",
        gt_sf="polymarket + kalshi + prediction market", gt_sea="polymarket + prediction market",
        hn=["prediction markets", "polymarket", "kalshi"], wiki="Prediction market",
        yc=r"prediction market|forecasting market", gh="prediction market", oa="prediction market"),
}

LABELLED = [k for k, v in SECTORS.items() if v["label"] in ("boom", "noise")]
CANDIDATES = [k for k, v in SECTORS.items() if v["label"] == "candidate"]

# World Bank indicators used for the SEA "readiness" layer
WB_INDICATORS = {
    "IT.NET.USER.ZS": "internet_users_pct",
    "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp",
    "SP.URB.TOTL.IN.ZS": "urban_pct",
    "FX.OWN.TOTL.ZS": "account_ownership_pct",
    "EG.ELC.ACCS.ZS": "electricity_access_pct",
    "SP.POP.TOTL": "population",
}
WB_COUNTRIES = ["SGP", "IDN", "VNM", "PHL", "THA", "MYS", "USA"]
