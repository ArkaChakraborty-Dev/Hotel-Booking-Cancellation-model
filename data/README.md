# Hotel Booking Cancellation Dataset

## Overview
This repository uses the Hotel Booking Demand dataset containing booking information for a City Hotel and a Resort Hotel, with bookings due to arrive between the 1st of July of 2015 and the 31st of August 2017.

- **Primary Archive**: `data/hotel_bookings.csv.zip`
- **Extracted File**: `data/hotel_bookings.csv`
- **Target Variable**: `is_canceled` (0 = Did not cancel, 1 = Canceled)

## Data Dictionary

| Variable | Description | Type |
|---|---|---|
| `hotel` | Hotel type (`City Hotel` or `Resort Hotel`) | Categorical |
| `is_canceled` | Target: whether booking was canceled (1) or not (0) | Binary |
| `lead_time` | Number of days between booking date and arrival date | Integer |
| `arrival_date_year` | Year of arrival | Integer |
| `arrival_date_month` | Month of arrival | Categorical |
| `arrival_date_week_number` | Week number of arrival date | Integer |
| `arrival_date_day_of_month` | Day of arrival date | Integer |
| `stays_in_weekend_nights` | Number of weekend nights (Sat/Sun) booked | Integer |
| `stays_in_week_nights` | Number of week nights (Mon-Fri) booked | Integer |
| `adults` | Number of adults | Integer |
| `children` | Number of children | Float |
| `babies` | Number of babies | Integer |
| `meal` | Type of meal booked (BB, HB, FB, SC, Undefined) | Categorical |
| `country` | Country of origin (ISO 3166-1 alpha-3 code) | Categorical |
| `market_segment` | Market segment designation (Online TA, Direct, etc.) | Categorical |
| `distribution_channel` | Booking distribution channel (TA/TO, Direct, etc.) | Categorical |
| `is_repeated_guest` | Whether the guest is a repeated guest (1) or not (0) | Binary |
| `previous_cancellations` | Prior canceled bookings by this customer | Integer |
| `previous_bookings_not_canceled` | Prior bookings not canceled by customer | Integer |
| `reserved_room_type` | Code of room type reserved | Categorical |
| `assigned_room_type` | Code of room type actually assigned *(Dropped as leakage)* | Categorical |
| `booking_changes` | Number of amendments made to booking *(Dropped as post-booking)* | Integer |
| `deposit_type` | Deposit policy (No Deposit, Non Refund, Refundable) | Categorical |
| `agent` | ID of travel agency that made booking | Categorical ID |
| `company` | ID of corporate entity that made/pays booking | Categorical ID |
| `days_in_waiting_list` | Days the booking was in waiting list before confirmation | Integer |
| `customer_type` | Type of booking (Transient, Contract, Transient-Party, Group) | Categorical |
| `adr` | Average Daily Rate (Revenue / total nights) | Float |
| `required_car_parking_spaces` | Number of car parking spaces requested | Integer |
| `total_of_special_requests` | Number of special requests made | Integer |
| `reservation_status` | Status (`Canceled`, `Check-Out`, `No-Show`) *(Dropped: 100% target leakage)* | Categorical |
| `reservation_status_date` | Date status was last updated *(Dropped: target leakage)* | Date |

## Anti-Leakage Rules Applied
To ensure real-world validity at booking creation time:
- `reservation_status` and `reservation_status_date` are strictly excluded as they directly reveal the outcome.
- `assigned_room_type` is determined at or near check-in, not at initial booking.
- `booking_changes` reflect modifications made after initial creation.
