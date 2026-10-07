"""Support-desk tickets for a practical routing demo.

Each row is one inbox item plus three labeled questions:

- queue: billing, shipping, access, or product
- escalate: a supervisor is needed now
- priority: Can wait, This week, Today, Right now

The ask is what the customer wants done. Other teams' words show up as
distractors. Option order is part of the input, so every row lists the queues
in the same order: billing, shipping, access, product.

Train and eval are different situations, not paraphrases of each other.
Demo tickets are a third set, stored without labels.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

QUEUES = {
    "billing": "Charges, invoices, refunds, taxes, and payment methods",
    "shipping": "Package location, delivery address, carriers, and items lost or damaged in transit",
    "access": "Login, password, two-factor, locked accounts, and sessions",
    "product": "The app or website: bugs, outages, and how a feature works",
}
LEVELS = ["Can wait", "This week", "Today", "Right now"]
QUEUE_INSTRUCTIONS = (
    "Which queue should own this ticket? "
    "Choose the team that can resolve what the customer is actually asking for. "
    "A mention of another team's words is not enough."
)
ESCALATE_INSTRUCTIONS = (
    "Does a supervisor need to take this now? "
    "Yes means a legal threat, an account compromise, a repeated failure after an earlier attempt, "
    "or a site-wide outage. A ticket can be urgent and still stay with the queue."
)
PRIORITY_INSTRUCTIONS = (
    "How soon does this customer need a response? "
    "Can wait means no deadline. This week means a normal service request. "
    "Today means they are blocked today. Right now means an incident in progress."
)

# queue, escalate, priority, channel, customer, subject, body
Case = tuple[str, bool, int, str, str, str, str]

TRAIN: list[Case] = [
    (
        "billing",
        False,
        1,
        "email",
        "Mina Shah",
        "Two charges for one order",
        "Order 4418 arrived yesterday and the tracking page says delivered. "
        "My Visa was charged $64 twice, three minutes apart. I only want the extra charge reversed. "
        "The parcel itself is fine.",
    ),
    (
        "billing",
        False,
        1,
        "chat",
        "Owen Blake",
        "Same sweater, two pending amounts",
        "Hi, the sweater is already on my table, so this is not a delivery problem. "
        "Checkout took $64 and then another $64. Please put one of them back on the card ending 4412.",
    ),
    (
        "billing",
        False,
        0,
        "email",
        "Helen Cho",
        "Invoices for the accountant",
        "Every shipment this quarter arrived. I do not need anyone to look up a parcel. "
        "Please send PDF invoices for January through March whenever finance picks them up. No deadline.",
    ),
    (
        "billing",
        False,
        0,
        "email",
        "Victor Alam",
        "Copies of receipts, nothing missing",
        "The packages are in the closet. I only need the receipts re-sent to billing@northwind.example "
        "so our bookkeeper can file them. Next month is soon enough.",
    ),
    (
        "billing",
        True,
        3,
        "email",
        "Ruth Keller",
        "My attorney files in the morning",
        "This is the formal notice. You billed my company twice for contract 77. "
        "If the duplicate $2,400 is not reversed before 9am, my attorney files with the state and the card network. "
        "I am not asking where a box is.",
    ),
    (
        "billing",
        True,
        3,
        "phone",
        "Chris Dalton",
        "Counsel is on the other line",
        "I'm calling with our lawyer. The second annual fee posted today. "
        "She wants a reversal confirmation while we are still on the call, or she sends the demand letter tonight.",
    ),
    (
        "billing",
        False,
        2,
        "chat",
        "Priya Nair",
        "Promo code never reduced the total",
        "I signed in fine, password was not the issue. Code SPRING took nothing off an order I placed an hour ago. "
        "The bank statement closes tonight, so I need the $18 difference back today.",
    ),
    (
        "billing",
        False,
        2,
        "email",
        "Elena Vos",
        "Charged full price after a valid code",
        "Login worked. The discount code was accepted on the summary and then the card was charged full price. "
        "Please correct the capture before end of day. I already have the item.",
    ),
    (
        "billing",
        False,
        1,
        "email",
        "Samuel Ortiz",
        "VAT number on the PDF is wrong",
        "The goods were delivered last Tuesday. The commercial invoice shows VAT 000 instead of our number ESB12345678. "
        "Our accountant cannot book it. Please reissue the invoice this week.",
    ),
    (
        "billing",
        False,
        1,
        "email",
        "Nora Feldman",
        "Reissue the tax invoice",
        "Do not reship anything. The crate arrived. The PDF lists a tax id we retired in 2022. "
        "I need a corrected invoice for the same payment.",
    ),
    (
        "billing",
        False,
        1,
        "phone",
        "Gabe Morin",
        "Renewal after I cancelled",
        "I cancelled the plan yesterday in the billing page. I was logged in. "
        "This morning the annual amount still posted. Please reverse that renewal. The shipment I am waiting on is a separate, wanted order.",
    ),
    (
        "billing",
        False,
        1,
        "email",
        "Lila Trent",
        "Stop the renewal charge",
        "Cancellation of the Plus plan is what I need reversed on the card. "
        "I can still see my tracking page, so access is fine, and the box can keep coming.",
    ),
    (
        "billing",
        True,
        2,
        "email",
        "Andre Wallace",
        "Refund approved twice, still absent",
        "Support approved this refund on the 2nd and again on the 9th. The card still shows the original charge. "
        "This is the third time I have asked. I need a supervisor to push the reversal today. The returned box was scanned in already.",
    ),
    (
        "billing",
        True,
        2,
        "chat",
        "June Park",
        "Third ask on the same refund",
        "I am not opening a new shipping question. You have the return. Two agents said the money was sent. "
        "It is not on the card. Please escalate, I have already done this twice.",
    ),
    (
        "billing",
        False,
        2,
        "chat",
        "Tomoko Abe",
        "Did the failed payment capture?",
        "Checkout said payment failed, but my bank shows a pending $90 hold. "
        "I need to know if you captured it before I retry, and I need that before 5pm. I never got a tracking number because nothing shipped.",
    ),
    (
        "billing",
        False,
        2,
        "email",
        "Hugh Baker",
        "Pending hold after an error page",
        "The site errored at the last step. My bank still placed a hold. "
        "Was the order paid or not? I have a meeting with finance today and need the answer before then.",
    ),
    (
        "billing",
        False,
        1,
        "email",
        "Sofia Marchetti",
        "Gift card balance is short",
        "I applied a $50 gift card. The confirmation only took $25 off, and the card balance now looks untouched. "
        "The product page and the delivery estimate were correct. Please fix the tender this week.",
    ),
    (
        "billing",
        False,
        1,
        "chat",
        "Ivan Petrov",
        "Gift card did not apply",
        "Not a login issue, I am clearly in my account. The gift card code was valid yesterday. "
        "Today's order ignored it. Can you apply the $50 to the charge?",
    ),
    (
        "billing",
        True,
        3,
        "phone",
        "Diane Cho",
        "Bank fraud desk is waiting",
        "I am on the phone with my bank's fraud team. A subscription I do not recognize posted an hour ago. "
        "They will not release my card until you confirm the charge is killed. Please do that while they are on the line.",
    ),
    (
        "billing",
        True,
        3,
        "email",
        "Mark Ellison",
        "Unrecognized subscription, bank on hold",
        "Fraud operations at my bank is in my inbox waiting for you. "
        "Kill the new $29 plan that appeared this morning and send a reversal reference. I can still log in. Nothing is in transit.",
    ),
    (
        "shipping",
        False,
        2,
        "email",
        "Carla Mendes",
        "Do not refund me, find the carton",
        "Invoice 2201 was correct and I already paid it. I am not asking for money back. "
        "Carton NL-2201 has sat in Columbus for five days with no scan. I need it moving today.",
    ),
    (
        "shipping",
        False,
        2,
        "chat",
        "Peter Lang",
        "Hub scan is five days old",
        "Please leave the charge alone. I want the box. Last scan was Columbus on Monday and it is now Saturday. "
        "Can the carrier be pushed today?",
    ),
    (
        "shipping",
        False,
        1,
        "chat",
        "Aisha Rahman",
        "Left at 4B, I live in 4D",
        "The delivery photo is apartment 4B. I am 4D, and the address on the paid invoice was right. "
        "Please have the carrier collect it. This week is fine if they still have it.",
    ),
    (
        "shipping",
        False,
        1,
        "email",
        "Ben Carter",
        "Neighbor received my order",
        "Payment is settled, do not touch it. The driver handed the parcel to 4B. "
        "My name is on 4D. I need a redelivery or a pickup from that neighbor.",
    ),
    (
        "shipping",
        False,
        0,
        "email",
        "Grace Kim",
        "Tracking later is fine",
        "This is a gift for a birthday next month. I do not need a refund and I am not locked out of anything. "
        "Send the tracking number whenever the label exists.",
    ),
    (
        "shipping",
        False,
        0,
        "email",
        "Ned Alvarez",
        "No rush on the label",
        "Ship it when the item is back in stock. I only want the tracking mail once you have it. "
        "There is no date I am trying to hit.",
    ),
    (
        "shipping",
        False,
        2,
        "phone",
        "Hannah Weiss",
        "Change the address before pickup",
        "I already paid. Please do not discuss the charge. The ship-to is my office, and I am home today. "
        "The truck has not picked up. Move it to 18 King Street before it leaves.",
    ),
    (
        "shipping",
        False,
        2,
        "chat",
        "Omar Farouk",
        "Hold the office delivery",
        "Paid in full. I need the destination swapped to my home before the carrier scans the label. "
        "I will be there all afternoon.",
    ),
    (
        "shipping",
        True,
        3,
        "email",
        "Emily Frost",
        "Rings are still in Memphis",
        "You promised overnight yesterday and missed it. The wedding is in six hours. "
        "The invoice is paid. I need a supervisor who can force a handoff, not another tracking link.",
    ),
    (
        "shipping",
        True,
        3,
        "phone",
        "Paul Igwe",
        "Ceremony is today, overnight failed",
        "This is the second promise that the garment would arrive this morning. It is still in Memphis. "
        "The event starts at 4. I need someone who can authorize a courier.",
    ),
    (
        "shipping",
        False,
        1,
        "email",
        "Rosa Delgado",
        "Vase arrived in pieces",
        "The invoice was paid and I do not want a speech about the product design. "
        "The outer carton was crushed and the vase is in shards. Please send a replacement this week.",
    ),
    (
        "shipping",
        False,
        1,
        "chat",
        "Kyle Brooks",
        "Transit damage, send another",
        "Box looked like it fell off a belt. Contents are broken. "
        "I want a replacement shipment, not a refund and not a website fix.",
    ),
    (
        "shipping",
        False,
        2,
        "chat",
        "Nina Volkov",
        "Only the left shoe was in the box",
        "I opened the parcel. One shoe. I will not accept store credit or a refund. "
        "Please send the missing shoe today. I paid the invoice already.",
    ),
    (
        "shipping",
        False,
        2,
        "email",
        "Felix Hartmann",
        "Pair arrived as a single",
        "The charge can stay. The right shoe never landed in the carton. "
        "I need the other one shipped out today.",
    ),
    (
        "shipping",
        False,
        2,
        "email",
        "Joan Pereira",
        "Marked delivered, lobby has nothing",
        "The charge can stay. Tracking says delivered to the concierge at 2:10 and the desk has no parcel. "
        "I fly out tomorrow morning, so I need a trace today.",
    ),
    (
        "shipping",
        False,
        2,
        "chat",
        "Samir Haddad",
        "Delivered scan, empty mailroom",
        "Do not reverse the payment. The carrier says the mailroom signed and the mailroom disagrees. "
        "Please open a trace. I leave town tomorrow.",
    ),
    (
        "shipping",
        True,
        2,
        "email",
        "Claire Dubois",
        "Sent to the old house again",
        "Last month I updated the address after you shipped to my old house. "
        "The new label has the old house again. This is the second time. I want a supervisor, and I want it intercepted today.",
    ),
    (
        "shipping",
        True,
        2,
        "phone",
        "Martin Boyle",
        "Address change did not stick",
        "I am calling because the corrected address was ignored a second time. "
        "Please escalate. The carton left this morning for the house I sold.",
    ),
    (
        "shipping",
        False,
        3,
        "email",
        "Yuki Tanaka",
        "Driver is at the gate",
        "The driver is outside right now and the gate code on the label is wrong. It is 4491, not 1944. "
        "I can see the van. I do not need a refund and my password works.",
    ),
    (
        "shipping",
        False,
        3,
        "chat",
        "Luis Ortega",
        "Van is waiting downstairs",
        "Carrier is at the curb. The entry code you printed is rejected. "
        "Correct code is 4491. They said they will leave in two minutes.",
    ),
    (
        "access",
        False,
        2,
        "email",
        "Rebecca Stone",
        "Reset mail never arrives",
        "I have an open invoice I am happy to pay once I can get in. That is not a dispute. "
        "The password reset has been requested four times and nothing reaches rebecca.stone@example. "
        "I need the contract downloaded today.",
    ),
    (
        "access",
        False,
        2,
        "chat",
        "Will Sanders",
        "No reset message after four tries",
        "The bill is fine, I am not writing about a charge. "
        "Reset emails are not showing up, including spam. I have to pull a file down before a call this afternoon.",
    ),
    (
        "access",
        False,
        1,
        "chat",
        "Fatima Diallo",
        "Locked after wrong passwords",
        "Too many tries and the account is locked. I know which password it should be now. "
        "Please unlock it this week. No package is involved.",
    ),
    (
        "access",
        False,
        1,
        "email",
        "George Hale",
        "Account lockout",
        "I locked myself out. Please clear the counter. "
        "I do not need a change to any order or invoice.",
    ),
    (
        "access",
        True,
        3,
        "phone",
        "Anita Desai",
        "Sign-in from a country I have never visited",
        "A session opened from Brazil ten minutes ago. I am in Chicago. Kill every session now. "
        "There is also a sweater in transit. Ignore the sweater.",
    ),
    (
        "access",
        True,
        3,
        "email",
        "Robert Klein",
        "Unknown country on the session list",
        "Please revoke all sessions immediately. Someone signed in from an IP I do not use. "
        "My outstanding delivery can wait. This cannot.",
    ),
    (
        "access",
        False,
        0,
        "email",
        "Chloe Martin",
        "Add a teammate next month",
        "Please create a login for jules@northwind.example. Jules starts next month, so there is no rush. "
        "This is not a billing change.",
    ),
    (
        "access",
        False,
        0,
        "email",
        "Patrick Owen",
        "New seat, no deadline",
        "We will need another user on the workspace eventually. "
        "Send an invite to jules@northwind.example whenever the queue is quiet.",
    ),
    (
        "access",
        True,
        3,
        "email",
        "Sandra Vogel",
        "Entire company is stuck in an SSO loop",
        "Every employee at Hart Co. hits an Okta loop since 8am. Nobody can reach the dashboard. "
        "This is an outage for our tenant. I need a supervisor on the identity side now. Orders we already paid can sit.",
    ),
    (
        "access",
        True,
        3,
        "phone",
        "Ian Brooks",
        "Okta bounce for the whole office",
        "All forty of us are locked in a sign-in loop. The site never reaches the home page. "
        "Please escalate to someone who can look at the SAML config today, actually right now.",
    ),
    (
        "access",
        False,
        1,
        "chat",
        "Maya Chen",
        "Site password fails in the app",
        "The same password works on the website. The iOS app says it is wrong. "
        "I can work on the website this week, but please fix the app login.",
    ),
    (
        "access",
        False,
        1,
        "email",
        "Elliot Marsh",
        "App rejects a good password",
        "I am in on the browser. The app will not take that password. "
        "No rush beyond this week, and I am not reporting a shipment.",
    ),
    (
        "access",
        False,
        2,
        "email",
        "Tara Singh",
        "Authenticator phone is gone",
        "I lost the phone that held the authenticator. I cannot print shipping labels from the dashboard until I get in. "
        "Those labels are my own outgoing orders, not a parcel you sent me. I need backup codes today.",
    ),
    (
        "access",
        False,
        2,
        "chat",
        "Hugo Berg",
        "Need backup codes before noon",
        "New phone, no authenticator app. I am blocked from the label screen. "
        "Please verify me and turn on backup codes today.",
    ),
    (
        "access",
        True,
        2,
        "email",
        "Paulina Krajewska",
        "Unlocked yesterday, locked again",
        "An agent cleared this lock yesterday afternoon. It locked again overnight with no tries from me. "
        "Please escalate. I have a customer call today and cannot reach the account.",
    ),
    (
        "access",
        True,
        2,
        "phone",
        "Steve Nolan",
        "Lock came back the next morning",
        "This is a repeat of yesterday's unlock. The account is frozen again. "
        "I need a supervisor because the first fix did not hold, and I am on a job site today.",
    ),
    (
        "access",
        False,
        1,
        "phone",
        "Irene Bakker",
        "Delete the profile",
        "The refund already landed last month. I am not calling about money. "
        "Please close the login and delete the profile. This week is fine.",
    ),
    (
        "access",
        False,
        1,
        "email",
        "Conrad Ellis",
        "Close the account",
        "Money questions are finished. Please remove my user and the saved address book. "
        "I do not need another receipt.",
    ),
    (
        "access",
        True,
        3,
        "chat",
        "Leila Haddad",
        "Kiosk still shows the last customer",
        "Your store kiosk let me see the previous person's orders and address. "
        "Kill that session now. I will not discuss my own package until that screen is cleared.",
    ),
    (
        "access",
        True,
        3,
        "email",
        "Noah Kim",
        "Shared tablet kept someone else's cart",
        "I picked up a floor tablet and it was signed in as another customer, with their address visible. "
        "Force a logout immediately and tell me it is done.",
    ),
    (
        "product",
        False,
        0,
        "email",
        "Janet Cole",
        "How do I export orders",
        "The package I bought arrived and I like it. This note is not about that parcel. "
        "Where in the app do I export orders to CSV? No rush at all.",
    ),
    (
        "product",
        False,
        0,
        "chat",
        "Boris Petrov",
        "CSV export location",
        "Quick question, no deadline. Which menu writes a CSV of past orders? "
        "I am not reporting a charge or a delivery.",
    ),
    (
        "product",
        False,
        2,
        "chat",
        "Amelia Grant",
        "Checkout button does nothing",
        "On my laptop the Pay button clicks and the page stays put. My card was not charged, so billing has nothing to reverse. "
        "I need to place the order today.",
    ),
    (
        "product",
        False,
        2,
        "email",
        "Diego Alvarez",
        "Pay control is dead",
        "The button does not submit. There is no charge and no shipment because the order never existed. "
        "Please fix the checkout action. I need it working today.",
    ),
    (
        "product",
        True,
        3,
        "email",
        "Christine Palmer",
        "Checkout spins for everyone",
        "Since noon the checkout page spins and never loads. Several customers have written. Cards are fine. "
        "This is a site outage. I need a supervisor on the product side now, not a refund for a personal order.",
    ),
    (
        "product",
        True,
        3,
        "phone",
        "Frank Meier",
        "Storefront checkout is down",
        "No customer can reach the payment step. The spinner never ends. "
        "Payments as a bank connection are not the complaint. The page is. Please escalate, sales are stopped.",
    ),
    (
        "product",
        False,
        1,
        "email",
        "Holly Jensen",
        "Search cannot find the blue mug",
        "The blue mug is in stock. I opened it by a direct link. Search for 'blue mug' returns nothing. "
        "I do not need a reship. Please fix search this week.",
    ),
    (
        "product",
        False,
        1,
        "chat",
        "Quinn Harper",
        "Catalog search misses a live SKU",
        "SKU MUG-BLUE loads if I paste the url and does not appear in search. "
        "The item is not lost in a warehouse. The search index is wrong.",
    ),
    (
        "product",
        False,
        1,
        "email",
        "Patricia Gomez",
        "Photo is red, title says green",
        "What you shipped matched the SKU, so leave the parcel alone. "
        "The product page photo is the red variant and the title says green. Please correct the page this week.",
    ),
    (
        "product",
        False,
        1,
        "chat",
        "Eddie Walsh",
        "Listing photo does not match the title",
        "Delivery was correct. The web page is not. Red photo, green title, same URL. "
        "I am asking for the content to be fixed.",
    ),
    (
        "product",
        False,
        2,
        "email",
        "Monica Breslin",
        "Dashboard skips today's orders",
        "The bank charge for order 88 is correct. Do not touch billing. "
        "The sales dashboard still omits every order placed today, and I close the books this afternoon.",
    ),
    (
        "product",
        False,
        2,
        "chat",
        "Alan Zhou",
        "Today's orders missing from the chart",
        "Charges match the bank. The analytics screen is the problem. "
        "Orders since midnight are absent. I need the chart fixed before I send numbers out today.",
    ),
    (
        "product",
        True,
        2,
        "email",
        "Jill Fraser",
        "Ticket 9001 is back",
        "The blank product editor from ticket 9001 has returned after yesterday's deploy. "
        "This is a repeat. My campaign starts today, so I need a supervisor, not the same restart advice.",
    ),
    (
        "product",
        True,
        2,
        "phone",
        "Seth Romero",
        "Editor bug returned after the deploy",
        "We already went through this on ticket 9001. The editor is blank again. "
        "Please escalate. The campaign cannot be edited and it starts today.",
    ),
    (
        "product",
        True,
        3,
        "email",
        "Vanessa Clark",
        "Prices render as zero",
        "Every price on the storefront shows $0 and people are placing orders. "
        "The payment service is not the bug I am reporting. The page is publishing zeros. Stop it now.",
    ),
    (
        "product",
        True,
        3,
        "chat",
        "Omar Said",
        "Storefront is giving stock away",
        "Prices display as zero across the catalog. Customers are checking out at no cost. "
        "This is live. Please get a product owner on it immediately.",
    ),
    (
        "product",
        False,
        1,
        "email",
        "Kelly Nguyen",
        "App says processing, the coat is on my chair",
        "Tracking was right and the coat is here. I am not asking the carrier for anything. "
        "The app still says Processing. Please correct the status. This week is fine.",
    ),
    (
        "product",
        False,
        1,
        "chat",
        "Brian Holt",
        "Status never moved to delivered",
        "I am wearing the jacket. The carrier finished. Your order screen did not. "
        "Fix the status display when you can this week.",
    ),
    (
        "product",
        False,
        2,
        "phone",
        "Sandra Iqbal",
        "Coupon box clears a valid code",
        "Billing confirmed code WELCOME is active and the amount is right. "
        "The field accepts it and then wipes it on blur, so customers cannot redeem it. Our campaign is today.",
    ),
    (
        "product",
        False,
        2,
        "email",
        "Chris Pappas",
        "Discount field erases the code",
        "The promotion itself is valid. Your billing colleague checked. "
        "The input clears itself, so the discount never reaches the order. We go live today.",
    ),
]

EVAL: list[Case] = [
    (
        "billing",
        False,
        1,
        "email",
        "Laura Bennett",
        "Resort fee that the contract waived",
        "The stay was fine and I am not reviewing the room. "
        "A $40 resort fee posted even though the signed contract waives it. Please take that fee off this week.",
    ),
    (
        "billing",
        False,
        0,
        "email",
        "Henry Zhao",
        "Proforma before we pay",
        "We will not ship anything until accounts payable has a proforma. "
        "Please email a proforma invoice for quote Q-19. There is no deadline this week.",
    ),
    (
        "billing",
        False,
        2,
        "chat",
        "Rita Osman",
        "New card is saved, please retry",
        "The renewal failed because the old card expired. I have saved the new one. "
        "Please retry the capture today so the plan does not lapse. I can already log in.",
    ),
    (
        "billing",
        False,
        1,
        "email",
        "Owen Shah",
        "Tax charged to an exempt org",
        "Our exemption certificate is attached. Sales tax was still added to invoice 551. "
        "The goods can stay where they are. Please reissue the invoice without tax.",
    ),
    (
        "billing",
        True,
        2,
        "email",
        "Marilyn Peck",
        "I will take the $12 fee to the attorney general",
        "A $12 convenience fee showed up with no mention at checkout. "
        "If it is not gone today I am filing with the attorney general and posting the receipt. I want a supervisor on this.",
    ),
    (
        "billing",
        False,
        0,
        "email",
        "Gabe Solomon",
        "Two receipt emails, one payment",
        "The parcel is on the porch, so delivery worked. I got two receipt emails and I am worried you will charge me twice. "
        "The card shows a single charge. Just confirm that when someone has a spare minute.",
    ),
    (
        "shipping",
        False,
        1,
        "email",
        "Nina Rossi",
        "PO box was rejected",
        "A refund is not what I asked for. The carrier rejected the PO box. "
        "Please send the parcel to the street address on file instead. This week is acceptable.",
    ),
    (
        "shipping",
        False,
        0,
        "chat",
        "Edith Klein",
        "Please require a signature",
        "No deadline. Add a signature requirement so the lobby does not take it. "
        "The payment already went through and I am not disputing it.",
    ),
    (
        "shipping",
        False,
        2,
        "email",
        "Malik Johnson",
        "Chocolate shipped without ice",
        "The cooler arrived warm and the chocolate is soft. I need an iced replacement today, before Saturday's party. "
        "I am not asking you to change the product page.",
    ),
    (
        "shipping",
        False,
        2,
        "phone",
        "Celia Ward",
        "Label still has the old zip",
        "The invoice total is right. The label was printed with my previous zip and the carrier has not picked up. "
        "Reprint it today.",
    ),
    (
        "shipping",
        True,
        2,
        "email",
        "Harvey Benton",
        "Third carrier handoff with no scan",
        "Two earlier tickets already covered missing handoff scans. It has happened again on the replacement. "
        "I want a supervisor to deal with the carrier today. Do not offer me a partial refund.",
    ),
    (
        "shipping",
        False,
        1,
        "chat",
        "Lucia Ferro",
        "Which bin did the doorman use",
        "The doorman signed. I need the bin or locker number so I can collect it after work. "
        "The charge is correct. This week, before Friday, is enough.",
    ),
    (
        "access",
        False,
        2,
        "email",
        "Jordan Hale",
        "IT rotated the password into spam",
        "Our IT desk rotated the password and the notice landed in spam, so I have been typing the old one. "
        "Please unlock and confirm the new password reached jordan.hale@example. I teach a class today.",
    ),
    (
        "access",
        False,
        0,
        "email",
        "Mei Lin",
        "Contractor needs an editor role",
        "A contractor starts next month and the read-only role is too weak. "
        "Please plan an editor invite. Nothing is broken tonight.",
    ),
    (
        "access",
        True,
        3,
        "chat",
        "Alex Romero",
        "I am in someone else's cart",
        "The website shows another person's name and their saved address in the cart I am looking at. "
        "Clear this session immediately. I do not want to see their data, and I am not placing an order.",
    ),
    (
        "access",
        False,
        1,
        "email",
        "Sophie Laurent",
        "Turn off two-factor on the old device",
        "I still have the backup email. Please let me remove the authenticator on a phone I no longer own. "
        "There is no fraud in progress. This week is fine.",
    ),
    (
        "access",
        False,
        2,
        "phone",
        "Daniel Cho",
        "Hotel wifi is blocked by the allow list",
        "I can prove it is me. The ip allow list is dropping the hotel connection and I present in an hour. "
        "Please add this network or give me a bypass today. This is not a stolen account.",
    ),
    (
        "access",
        False,
        0,
        "email",
        "Emma Brooks",
        "Username after a name change",
        "I would like my username updated after a name change. "
        "I also want a copy of the last invoice later, but that is not the request in this email. No rush.",
    ),
    (
        "product",
        False,
        1,
        "email",
        "Hassan Ali",
        "Color filter returns an empty page",
        "Filtering the catalog to green returns no products, while the green items are visible in the full list. "
        "I am not waiting on a parcel. Please repair the filter this week.",
    ),
    (
        "product",
        True,
        3,
        "email",
        "Catherine Moore",
        "Confirmation page shows the previous customer",
        "After payment the confirmation screen showed a stranger's first name. "
        "I completed my own purchase. This is a product bug leaking another customer. Please treat it as an incident now.",
    ),
    (
        "product",
        False,
        0,
        "chat",
        "Leo Martins",
        "How do I duplicate a listing",
        "Is there a duplicate action on a listing? I am setting up the catalog slowly and there is no deadline. "
        "Please point me at the control.",
    ),
    (
        "product",
        False,
        2,
        "phone",
        "Ruth Okonkwo",
        "Scanner field loses focus",
        "In the receiving screen the barcode box blurs after every scan, so the warehouse cannot receive today's freight. "
        "The freight is not a customer shipment complaint. The screen is. We need it today.",
    ),
    (
        "product",
        False,
        1,
        "email",
        "Simon Adler",
        "Store map is a blank tile",
        "The iOS app opens a blank map where store locations should be. The web map works. "
        "I will visit next week, so this week is soon enough.",
    ),
    (
        "product",
        False,
        2,
        "email",
        "Naomi Cohen",
        "Scheduled publish did not run",
        "The campaign went live with yesterday's prices because the scheduled publish never fired. "
        "Card charges on the old orders are a different system and they are correct. I need the publisher fixed today. First time this has happened.",
    ),
]

DEMO: list[Case] = [
    (
        "billing",
        False,
        1,
        "email",
        "Alice Berger",
        "Extra capture on order 7781",
        "Order 7781 was delivered this morning, tracking and all. I am writing only because the card was captured twice for $128. "
        "Please reverse one capture. I do not need a new shipment.",
    ),
    (
        "product",
        True,
        3,
        "email",
        "Marcus Webb",
        "No one can complete checkout",
        "For the last 40 minutes every checkout on the public site spins and fails. "
        "Individual refunds are not the issue. The payment page itself is down and the register is silent. Please get someone senior on the outage.",
    ),
    (
        "access",
        False,
        2,
        "chat",
        "Jill Kramer",
        "Reset link is missing",
        "I have a parcel due tomorrow and I am not worried about it. "
        "I cannot sign in. The reset link has not arrived after three requests, and I need the account today to change a ship-to on a different system.",
    ),
    (
        "shipping",
        False,
        2,
        "email",
        "Tony Rinaldi",
        "Leave the charge, send the crate",
        "Please do not refund order 3004. I want the crate. It has not scanned out of the origin facility in four days. "
        "Push the carrier today.",
    ),
    (
        "billing",
        True,
        3,
        "phone",
        "Susan Clarke",
        "Demand letter goes out at noon",
        "Our counsel told me to call. The duplicate implementation fee is still on the card. "
        "If we do not have a reversal reference by noon, the demand letter goes to your registered agent.",
    ),
    (
        "billing",
        False,
        0,
        "email",
        "Peter Walsh",
        "Receipt whenever you have a minute",
        "Could someone resend the March receipt to my accountant? The hoodie arrived long ago. "
        "There is no problem and no date attached.",
    ),
    (
        "shipping",
        False,
        1,
        "email",
        "Maria Santos",
        "Frame arrived with a broken corner",
        "The charge is correct and the listing photo was accurate. The frame's corner was smashed in transit. "
        "Please ship a replacement this week.",
    ),
    (
        "access",
        True,
        3,
        "email",
        "Dev Patel",
        "Staff cannot sign in and the sale starts at 6",
        "Every staff login is rejected and the doorbuster starts at 6pm. "
        "This is the second outage of the identity provider today. I need a supervisor, not a password tip.",
    ),
    (
        "product",
        False,
        1,
        "chat",
        "Erin Blake",
        "Screen still says label created",
        "The shoes are on my floor. The carrier finished yesterday. "
        "Your order page still says label created. I am asking you to fix the page, not the carrier. This week is fine.",
    ),
    (
        "shipping",
        False,
        3,
        "phone",
        "John Abebe",
        "Courier is in the lobby",
        "The courier is in the lobby and the suite number on the label is 1200. We are 1204. "
        "They will not come up without a corrected number. I am standing with them.",
    ),
]


def questions(case: Case, *, labeled: bool) -> dict:
    queue, escalate, priority, _channel, _customer, _subject, _body = case
    queue_question: dict = {
        "type": "choice",
        "instructions": QUEUE_INSTRUCTIONS,
        "criteria": dict(QUEUES),
    }
    escalate_question: dict = {
        "type": "noul",
        "instructions": ESCALATE_INSTRUCTIONS,
        "criteria": {
            "false": "The owning queue can handle it",
            "true": "A supervisor should take it now",
        },
    }
    priority_question: dict = {
        "type": "score",
        "instructions": PRIORITY_INSTRUCTIONS,
        "criteria": list(LEVELS),
    }
    if labeled:
        queue_question["label"] = queue
        escalate_question["label"] = escalate
        priority_question["label"] = priority
    return {"queue": queue_question, "escalate": escalate_question, "priority": priority_question}


def row(case: Case, *, labeled: bool) -> dict:
    _queue, _escalate, _priority, channel, customer, subject, body = case
    return {
        "state": {
            "channel": channel,
            "customer": customer,
            "subject": subject,
            "body": body,
        },
        "questions": questions(case, labeled=labeled),
    }


def rows_from(cases: list[Case], *, labeled: bool, seed: int | None = None) -> list[dict]:
    built = [row(case, labeled=labeled) for case in cases]
    if seed is None:
        return built
    shuffled = list(built)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _bodies(cases: list[Case]) -> list[str]:
    return [case[-1] for case in cases]


def validate() -> None:
    allowed = set(QUEUES)
    for name, cases in (("train", TRAIN), ("eval", EVAL), ("demo", DEMO)):
        if not cases:
            raise ValueError(f"{name} is empty")
        for case in cases:
            queue, escalate, priority, channel, customer, subject, body = case
            if queue not in allowed:
                raise ValueError(f"{name}: unknown queue {queue}")
            if not isinstance(escalate, bool):
                raise ValueError(f"{name}: escalate must be bool")
            if priority not in {0, 1, 2, 3}:
                raise ValueError(f"{name}: priority {priority}")
            if channel not in {"email", "chat", "phone"}:
                raise ValueError(f"{name}: channel {channel}")
            if not all(part.strip() for part in (customer, subject, body)):
                raise ValueError(f"{name}: empty field on {subject!r}")
    train_bodies = _bodies(TRAIN)
    eval_bodies = _bodies(EVAL)
    demo_bodies = _bodies(DEMO)
    for label, bodies in (("train", train_bodies), ("eval", eval_bodies), ("demo", demo_bodies)):
        if len(set(bodies)) != len(bodies):
            raise ValueError(f"{label} repeats a body")
    overlap = set(train_bodies) & set(eval_bodies) | set(train_bodies) & set(demo_bodies) | set(eval_bodies) & set(demo_bodies)
    if overlap:
        raise ValueError("train, eval, and demo share a body")
    for queue in allowed:
        if not any(case[0] == queue for case in TRAIN):
            raise ValueError(f"train has no {queue}")
        if not any(case[0] == queue for case in EVAL):
            raise ValueError(f"eval has no {queue}")
    if not any(case[1] for case in TRAIN) or all(case[1] for case in TRAIN):
        raise ValueError("train escalate labels do not vary")
    if len({case[2] for case in TRAIN}) < 4:
        raise ValueError("train is missing a priority level")


def _write_jsonl(path: Path, payload: list[dict]) -> None:
    lines = [json.dumps(item, ensure_ascii=True) for item in payload]
    path.write_text("\n".join(lines) + "\n")


def write_support(root: Path | None = None) -> Path:
    validate()
    destination = root or Path("data/support")
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / "train.jsonl", rows_from(TRAIN, labeled=True, seed=0))
    _write_jsonl(destination / "eval.jsonl", rows_from(EVAL, labeled=True, seed=1))
    demo_dir = destination / "demo"
    demo_dir.mkdir(exist_ok=True)
    expected = []
    for index, case in enumerate(DEMO, start=1):
        slug = "".join(char if char.isalnum() else "-" for char in case[5].lower())
        while "--" in slug:
            slug = slug.replace("--", "-")
        filename = f"{index:02d}-{slug.strip('-')[:40].strip('-')}.json"
        (demo_dir / filename).write_text(json.dumps(row(case, labeled=False), indent=2) + "\n")
        expected.append(
            {
                "file": filename,
                "queue": case[0],
                "escalate": case[1],
                "priority": case[2],
            }
        )
    (demo_dir / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    return destination


def main() -> None:
    root = write_support()
    print(f"wrote {root}")
    print(f"train {len(TRAIN)}  eval {len(EVAL)}  demo {len(DEMO)}")


if __name__ == "__main__":
    main()
