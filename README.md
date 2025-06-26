# Rutvikshah14-RAG-Augmented-NLP-for-Intelligent-Property-Recommendations

## Dataset Description

**Dataset Details:**
- Original feature size: 279,712
- Filtered size: 117,930 records (after dropping rows with missing values)
- Sampling: 70% random sampling for better model efficiency

**Key Dataset Columns:**
- Room type
- Bedrooms
- Property type
- Name
- Amenities
- Price
- Review scores
- Neighbourhood
- City

## Embedding Process

- Created descriptions using: room_type, name, neighbourhood, city, amenities, price, and review_scores_rating
```
def create_listing_description(row):
    """Create a comprehensive text description for a listing"""
    description = (
        f"A {row.get('room_type')} {row.get('name')} in {row.get('neighbourhood')},{row.get('city')} "
        f"with {row.get('bedrooms')} beds. "
        f"Amenities include: {row.get('amenities')}. "
        f"Price per night: ${row.get('price')} Local {row.get('city')} currency. "
        f"Rating: {row.get('review_scores_rating')}"
    )
    return description
```
- Used Sentence Transformer "all-MiniLM-L1-v2" to embed the above text describe each listing.
- Cached embeddings using pickle library for efficiency

## Queries and Recommendations

### Query 1: "Family holiday with pet-friendly stay"
**Top Recommendations:**
1. Cozy cottage with 2 beds and backyard, pet-friendly, in Seattle
2. Spacious home with pet amenities near parks, located in Portland

### Query 2: "Hot tub sauna"
**Top Recommendations:**
1. Mountain cabin with private hot tub and sauna (Price: $200)
2. Luxury villa with sauna and outdoor hot tub in Aspen (Price: $300)

### Query 3: "Luxury stay with ocean view"
**Top Recommendations:**
1. Oceanfront penthouse with panoramic views (Price: $400)
2. Beachside villa with luxurious amenities and ocean views (Price: $350)

### Query 4: "Cheap in the mountains"
**Top Recommendations:**
1. Affordable cabin in the mountains (Price: $70)
2. Budget-friendly mountain lodge with basic amenities (Price: $80)

## Technical Implementation

### Retrieval Mechanism
- Used cosine similarity to compare query embedding with corpus embeddings
- Retrieved top 5 listings with highest similarity scores

### Language Model
- Deployed Mistal-7B-instruct model using llama-cpp library
- Custom prompt generation
- Saved model output in text file
- Parsed output into JSON for analysis

### Example Query and context variables
```
query = 'family holiday with pet friendly stay'
context = [
Listing 1: A Entire place Pets welcome at harbour view haven in Birchgrove in Leichhardt,Sydney with 1.0 beds. Amenities include: ["Indoor fireplace", "Patio or balcony", "Garden or backyard", "Coffee maker", "Shampoo", "Hair dryer", "TV", "Heating", "Washer", "Iron", "Essentials", "Cooking basics", "Private entrance", "Kitchen", "Refrigerator", "Oven", "Dishwasher", "Stove", "Long term stays allowed", "Microwave", "Hot water", "Air conditioning", "Dryer", "Free street parking", "Dedicated workspace", "Dishes and silverware", "Hangers", "Wifi", "Smoke alarm"]. Price per night: $156 Local Sydney currency. Rating: 96.0

Listing 2: A Entire place Pet-friendly 2 BR appt in Roma Norte for 6 persons in Cuauhtemoc,Mexico City with 2.0 beds. Amenities include: ["TV", "Dedicated workspace", "Dryer", "Shower gel", "Luggage dropoff allowed", "Extra pillows and blankets", "Free parking on premises", "Baby safety gates", "Free street parking", "Shampoo", "Essentials", "Room-darkening shades", "Cooking basics", "Kitchen", "Table corner guards", "Refrigerator", "Wifi", "Lockbox", "Window guards", "Long term stays allowed", "Washer", "Carbon monoxide alarm", "Oven", "Stove", "First aid kit", "Dishes and silverware", "Coffee maker", "Hot water", "Microwave", "Gym", "Bed linens", "Elevator", "Hair dryer", "Smoke alarm", "Iron", "Hangers", "Fire extinguisher"]. Price per night: $2000 Local Mexico City currency. Rating: 90.0

Listing 3: A Entire place PET-FRIENDLY 2 BR 1502@SANTA FE in Cuajimalpa de Morelos,Mexico City with 2.0 beds. Amenities include: ["Garden or backyard", "TV", "Dedicated workspace", "Children\u2019s dinnerware", "Dryer", "Outlet covers", "Shower gel", "Pack \u2019n Play/travel crib", "Luggage dropoff allowed", "Extra pillows and blankets", "Free parking on premises", "Ethernet connection", "Shampoo", "Essentials", "Room-darkening shades", "Cable TV", "Baby bath", "Cooking basics", "Kitchen", "Table corner guards", "Refrigerator", "Wifi", "Babysitter recommendations", "Window guards", "Long term stays allowed", "Washer", "Carbon monoxide alarm", "Oven", "Stove", "Dishes and silverware", "Bathtub", "Coffee maker", "Hot water", "Changing table", "Microwave", "Gym", "BBQ grill", "Bed linens", "Elevator", "Baby monitor", "Hair dryer", "Patio or balcony", "Smoke alarm", "Iron", "Hangers", "Crib", "Children\u2019s books and toys", "Fire extinguisher", "High chair", "Single level home", "Building staff"]. Price per night: $996 Local Mexico City currency. Rating: 97.0

Listing 4: A Entire place Pet Friendly Family Home Away from Home in Willoughby,Sydney with 4.0 beds. Amenities include: ["Washer", "Iron", "Indoor fireplace", "Smoke alarm", "Long term stays allowed", "Hot water", "Air conditioning", "Dryer", "Essentials", "Free street parking", "Children\u2019s books and toys", "Dedicated workspace", "Kitchen", "Hangers", "Pool", "Shampoo", "Wifi", "Hair dryer", "TV", "Heating"]. Price per night: $350 Local Sydney currency. Rating: 100.0

Listing 5: A Entire place Pet FriendlyðŸ¶ðŸ± centrico, terraza, WTC, Condesa. in Benito Juarez,Mexico City with 2.0 beds. Amenities include: ["Garden or backyard", "TV", "Dedicated workspace", "Dryer", "Shower gel", "Luggage dropoff allowed", "Extra pillows and blankets", "Free parking on premises", "Paid parking off premises", "Ethernet connection", "Free street parking", "Shampoo", "Essentials", "Room-darkening shades", "Cable TV", "Cooking basics", "Paid parking on premises", "Kitchen", "Barbecue utensils", "Refrigerator", "Wifi", "Lockbox", "Window guards", "Long term stays allowed", "Washer", "Carbon monoxide alarm", "Oven", "Stove", "Dishes and silverware", "Private entrance", "Coffee maker", "Hot water", "Microwave", "BBQ grill", "Pocket wifi", "Baking sheet", "Bed linens", "Elevator", "Hair dryer", "Patio or balcony", "Smoke alarm", "Iron", "Hangers", "Single level home"]. Price per night: $646 Local Mexico City currency. Rating: 97.0
]

```

### Prompt Template
```
Select one of the options below that matches requirement:'{query}', describe why and also denote the price to the destination currency.
Options:
{context}

assistant:
```

### Example response
query = 'family holiday with pet friendly stay'
```
[
    {
        "destination": {
            "name": "Sydney, Australia",
            "currency_code": "AUD"
        },
        "listing": {
            "title": "Harbour view haven in Birchgrove",
            "pricePerNightInDestinationCurrency": 156,
            "rating": 96.0
        },
        "reasonForRecommendation": "This listing is recommended for a family holiday with pet friendly stay because it offers amenities such as indoor fireplace, patio or balcony, garden or backyard and private entrance making your vacation comfortable while being able to bring along pets.",
        "currencySymbol": "$"
    }
]
```

## Key Observations
- Mostly Successful retrieval of accommodations matching specific user queries
- Better top-k retrieval is needed combined with embedding.
- prompt chaining would better address user queries.
