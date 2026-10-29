"""General, educational information for each condition the model was trained on.

This is general health-literacy content, not personalised medical advice. It deliberately
avoids drug names, doses and outcome guarantees: treatment decisions belong to a clinician.
"""

from __future__ import annotations

GENERAL_ROUTINE = {
    "daily": [
        "Wash with a mild, fragrance-free shampoo as often as your scalp needs (oily scalps may need it daily).",
        "Rinse thoroughly. Shampoo or conditioner residue can irritate the scalp.",
        "Pat hair dry, or use a hair dryer on a low-heat setting.",
        "Detangle gently with a wide-tooth comb, starting from the ends.",
        "Protect your scalp from strong sun with a hat, especially where hair is thin.",
    ],
    "weekly": [
        "Check your scalp in good light (or ask someone) for new patches, redness or flaking.",
        "Wash combs, brushes, pillowcases and hats.",
        "Give your hair a break from tight hairstyles, heat styling and chemical treatments.",
        "Take a photo of the same area in the same lighting, so you can compare honestly over time.",
    ],
    "nutrition": [
        "Eat a varied diet with enough protein (eggs, legumes, fish, dairy, lean meat or tofu). Hair is built from protein.",
        "Include iron- and zinc-rich foods (leafy greens, legumes, nuts, seeds, meat). Low iron is a common, "
        "treatable contributor to hair shedding.",
        "Get omega-3 fats (oily fish, walnuts, flax) and plenty of fruit and vegetables.",
        "Stay hydrated and avoid crash diets. Rapid weight loss can trigger hair shedding.",
        "Take supplements only after a blood test and medical advice. Excess vitamin A or selenium can itself cause hair loss.",
    ],
}

RED_FLAGS = [
    "Sudden or rapid hair loss, or bald patches that keep spreading",
    "Pain, swelling, pus, oozing, or a scalp that is hot to the touch",
    "Fever or feeling unwell alongside scalp symptoms",
    "Scarring, shiny skin, or areas where hair does not grow back",
    "Itching or rash that disturbs sleep or does not improve after 2–4 weeks of gentle care",
    "Symptoms in a child, or several people in your household affected",
]

CONDITIONS: dict[str, dict] = {
    "alopecia_areata": {
        "name": "Alopecia Areata",
        "category": "Hair loss · autoimmune",
        "summary": (
            "An autoimmune condition in which the immune system targets hair follicles. It typically causes "
            "smooth, round or oval bald patches, and is not caused by poor hygiene."
        ),
        "signs": ["Smooth, round bald patches", "Short broken hairs at the patch edges", "Sometimes nail pitting"],
        "care": [
            "Be gentle with the affected area. Avoid harsh products and tight hairstyles.",
            "Use sun protection on exposed scalp patches.",
            "Track patch size with photos taken in the same conditions.",
            "Look for peer support. Hair loss can affect wellbeing, and support groups exist.",
        ],
        "expectations": (
            "The course is unpredictable. Some patches regrow on their own over months, and new patches can appear. "
            "A dermatologist can discuss treatments that may speed regrowth. Responses vary from person to person."
        ),
        "urgency": "Recommended",
        "doctor": "See a dermatologist to confirm the diagnosis and discuss treatment options. Go promptly if patches spread quickly.",
    },
    "contact_dermatitis": {
        "name": "Contact Dermatitis",
        "category": "Inflammation · irritant / allergic",
        "summary": (
            "Skin inflammation caused by something that touched the scalp, often a hair dye, a fragrance, a "
            "preservative or a styling product. It can be an irritant reaction or an allergy."
        ),
        "signs": ["Redness and itching", "Burning or stinging", "Rash that may spread to the forehead, ears or neck"],
        "care": [
            "Stop using any recently introduced product, especially hair dye.",
            "Switch to a fragrance-free, hypoallergenic shampoo.",
            "Rinse the scalp with lukewarm, not hot, water.",
            "Patch-test new products on a small area of skin for 48 hours before using them on your scalp.",
        ],
        "expectations": (
            "Once the trigger is avoided, mild reactions often settle within one to a few weeks. "
            "Identifying the exact allergen may need patch testing by a dermatologist."
        ),
        "urgency": "If severe or persistent",
        "doctor": "See a doctor urgently for facial or eye swelling, blistering or oozing. Otherwise, see a dermatologist if it lasts more than 2 weeks or keeps coming back.",
    },
    "folliculitis": {
        "name": "Folliculitis",
        "category": "Inflammation · follicles",
        "summary": (
            "Inflammation of the hair follicles, often caused by bacteria or yeast, friction or blocked follicles. "
            "It usually looks like small red bumps or pus-filled spots around the hairs."
        ),
        "signs": ["Small red bumps or whiteheads around follicles", "Tenderness or itching", "Crusting"],
        "care": [
            "Keep the scalp clean and avoid scratching or picking at the spots.",
            "Avoid shaving the scalp, very tight headwear and heavy oils while it is flaring.",
            "Wash pillowcases, towels and hats regularly. Do not share them.",
            "Apply a warm (not hot) compress to ease discomfort.",
        ],
        "expectations": (
            "Mild superficial cases often improve within days to a couple of weeks with good hygiene. "
            "Deeper or recurring cases may need medical treatment."
        ),
        "urgency": "If spreading or painful",
        "doctor": "See a doctor for large, painful or spreading bumps, fever, scarring, or if it keeps coming back.",
    },
    "head_lice": {
        "name": "Head Lice",
        "category": "Infestation · contagious",
        "summary": (
            "Tiny insects that live on the scalp and lay eggs (nits) on hair shafts. They are very common, especially "
            "in children, spread by head-to-head contact, and are not a sign of poor hygiene."
        ),
        "signs": ["Itching, especially behind the ears and at the nape", "Nits stuck firmly near the hair roots", "Moving lice"],
        "care": [
            "Wet-comb conditioned hair with a fine-toothed lice comb, section by section.",
            "Use a lice treatment from a pharmacist and follow the label exactly. Most need a second application.",
            "Check everyone in the household and treat only those with live lice.",
            "Wash recently used bedding, hats and brushes in hot water.",
        ],
        "expectations": (
            "Thorough treatment plus repeat combing usually clears lice within about 2 weeks. Itching can persist "
            "for a while afterwards. Re-infestation is common if contacts are not checked."
        ),
        "urgency": "If treatment fails",
        "doctor": "Ask a pharmacist or doctor if lice persist after two correct treatments, the scalp looks infected, or the person is an infant.",
    },
    "lichen_planus": {
        "name": "Lichen Planus (Lichen Planopilaris)",
        "category": "Inflammation · scarring risk",
        "summary": (
            "An inflammatory condition that, on the scalp (lichen planopilaris), can damage hair follicles and "
            "cause permanent, scarring hair loss if untreated."
        ),
        "signs": ["Redness and scaling around follicles", "Itching or burning", "Patches of thinning, sometimes shiny skin"],
        "care": [
            "Use gentle, fragrance-free products and avoid harsh scrubbing.",
            "Avoid tight hairstyles and chemical treatments.",
            "Photograph the affected area regularly to monitor change.",
        ],
        "expectations": (
            "Treatment aims to stop progression. Hair lost to scarring usually does not regrow, which makes early "
            "assessment important. The condition can be long-lasting."
        ),
        "urgency": "Recommended soon",
        "doctor": "See a dermatologist promptly. Early treatment can help prevent permanent hair loss, and a biopsy may be needed to confirm the diagnosis.",
    },
    "male_pattern_baldness": {
        "name": "Male Pattern Baldness (Androgenetic Alopecia)",
        "category": "Hair loss · genetic / hormonal",
        "summary": (
            "A very common, gradual hair thinning influenced by genetics and hormones. It typically affects the temples "
            "and crown in men, and causes diffuse thinning at the parting in women."
        ),
        "signs": ["Receding hairline or thinning crown", "Finer, shorter hairs over time", "Gradual progression over years"],
        "care": [
            "Treat your hair gently: low heat, no tight styles, careful brushing.",
            "Protect the thinning scalp from the sun.",
            "Take consistent photos every 3 months to judge change objectively.",
            "Be wary of products with miracle claims. Only a few treatments have good evidence.",
        ],
        "expectations": (
            "Without treatment it usually progresses slowly. Evidence-based treatments, discussed with a doctor, may "
            "slow thinning, and some people see partial regrowth after 6–12 months of consistent use. Results vary."
        ),
        "urgency": "Optional",
        "doctor": "See a doctor or dermatologist to discuss evidence-based options, or if the loss is sudden, patchy, or comes with other symptoms.",
    },
    "psoriasis": {
        "name": "Scalp Psoriasis",
        "category": "Inflammation · chronic, autoimmune",
        "summary": (
            "A long-term immune-related skin condition that causes thick, well-defined red plaques with silvery-white "
            "scale. It can extend beyond the hairline and is not contagious."
        ),
        "signs": ["Raised red plaques with silvery scale", "Itching or soreness", "Scale extending past the hairline"],
        "care": [
            "Soften scale before washing, and remove it gently. Do not pick or scratch it off.",
            "Ask a pharmacist about over-the-counter medicated shampoos (for example coal tar or salicylic acid).",
            "Keep the skin moisturised, and manage known triggers such as stress, smoking and alcohol.",
        ],
        "expectations": (
            "Psoriasis tends to come and go in flares. With a suitable treatment plan, many people achieve good "
            "control over weeks, but it usually needs ongoing management rather than a one-time cure."
        ),
        "urgency": "Recommended",
        "doctor": "See a doctor or dermatologist for a diagnosis and treatment plan, especially if it is extensive, or if you have joint pain (which can indicate psoriatic arthritis).",
    },
    "seborrheic_dermatitis": {
        "name": "Seborrheic Dermatitis",
        "category": "Inflammation · yeast-related",
        "summary": (
            "A common inflammatory condition linked to skin oil and the natural yeast on the skin. Mild forms are "
            "known as dandruff. It causes greasy, yellowish or white flakes and redness."
        ),
        "signs": ["Greasy yellow or white flakes", "Redness and itching", "May also affect the eyebrows, nose folds and ears"],
        "care": [
            "Use an anti-dandruff shampoo (for example zinc pyrithione, ketoconazole or selenium sulfide) and leave it on for a few minutes before rinsing.",
            "Rotate between two types of anti-dandruff shampoo if one stops working.",
            "Wash regularly. Letting oil build up tends to worsen it.",
            "Manage stress and sleep. Flares are often linked to both.",
        ],
        "expectations": (
            "Symptoms often improve noticeably within 2–4 weeks of regular medicated shampoo use. It tends to "
            "return, so maintenance washing once or twice a week is common."
        ),
        "urgency": "If not improving",
        "doctor": "See a doctor if there is no improvement after 4 weeks of an anti-dandruff routine, or if it is severe, painful or spreading.",
    },
    "telogen_effluvium": {
        "name": "Telogen Effluvium",
        "category": "Hair loss · shedding",
        "summary": (
            "Temporary, diffuse hair shedding that typically starts 2–3 months after a trigger such as illness, "
            "fever, surgery, childbirth, severe stress, crash dieting or certain medicines."
        ),
        "signs": ["Increased hair fall when washing or brushing", "Diffuse thinning rather than bald patches", "Hairline usually intact"],
        "care": [
            "Think back 2–3 months for a possible trigger.",
            "Focus on balanced nutrition and adequate protein.",
            "Be gentle with your hair. Shedding hairs come out easily, but this does not damage the follicle.",
            "Manage stress and sleep. Avoid unproven 'hair growth' products.",
        ],
        "expectations": (
            "Once the trigger resolves, shedding often slows within about 3–6 months, and visible regrowth follows "
            "over the next months. Ongoing shedding beyond 6 months deserves a check."
        ),
        "urgency": "If persistent",
        "doctor": "See a doctor if shedding lasts more than 6 months, if you have symptoms such as tiredness or weight change (blood tests may help), or if bald patches appear.",
    },
    "tinea_capitis": {
        "name": "Tinea Capitis (Scalp Ringworm)",
        "category": "Infection · fungal, contagious",
        "summary": (
            "A fungal infection of the scalp and hair shafts, most common in children. It can cause scaly patches, "
            "broken hairs and hair loss, and it spreads by contact."
        ),
        "signs": ["Scaly, itchy patches", "Broken hairs or black dots", "Sometimes swollen, tender areas"],
        "care": [
            "Do not share combs, brushes, hats, towels or pillows.",
            "Wash and disinfect hair tools and bedding.",
            "Medicated shampoo may reduce spread but does not cure it on its own.",
        ],
        "expectations": (
            "It usually needs prescription treatment taken by mouth for several weeks, and hair typically regrows "
            "after the infection clears. Without proper treatment it rarely resolves."
        ),
        "urgency": "Recommended soon",
        "doctor": "See a doctor. Scalp ringworm usually needs prescription treatment, and close contacts may need checking. Go urgently if a boggy, painful swelling develops.",
    },
}
