# Invoice template — fmm-fairness-eval

```
INVOICE
=======

Invoice number:   FMM_FAIRNESS_EVAL_CLI-{YYYY}-{NNN}
Issue date:       {DD MMMM YYYY}
Due date:         {Issue date + 14 days, or "on receipt" for subscription auto-charge}
Payment terms:    Subscription tiers: auto-charged via Stripe on renewal.
                  One-time engagements: 50% advance, 50% on delivery.
                  Net 14 days from issue otherwise.

------------------------------------------------------------------
ISSUER (Service provider)
------------------------------------------------------------------
Name:             César Pereiro García
Status:           Individual professional (autónomo registration {pending / NIF})
Tax ID (NIF):     {NIF}
Address:          {Address, postcode, city, Spain}
Email:            cesar.pereiro.garcia@gmail.com
Phone:            {Phone, optional}

------------------------------------------------------------------
RECIPIENT (Customer)
------------------------------------------------------------------
Legal name:       {Buyer legal entity}
Tax ID / VATIN:   {EU VATIN, e.g. DE123456789}
Address:          {Buyer registered address}
Contact:          {Buyer contact name, role}
Email:            {Buyer billing email}

------------------------------------------------------------------
LINE ITEMS
------------------------------------------------------------------
Description                                   Qty    Unit (EUR)   Total (EUR)
fmm-fairness-eval {tier name} subscription,
  billing period {YYYY-MM-DD to YYYY-MM-DD}.        1      {Amount}       {Amount}

------------------------------------------------------------------
TOTALS
------------------------------------------------------------------
Subtotal:                                                          {Amount}
VAT:                                                               0.00
  Reason for 0% VAT:
    Option 1 (EU B2B reverse-charge, default):
      "Reverse-charge applicable per Article 196 of Council
       Directive 2006/112/EC. Recipient is liable for VAT in
       Member State of establishment."
    Option 2 (Non-EU buyer):
      "Out of scope of EU VAT — supply to recipient established
       outside the European Union."
    Option 3 (Spanish buyer requiring IVA):
      "Engagement deferred to autónomo registration completion."

TOTAL DUE (EUR):                                                   {Amount}

------------------------------------------------------------------
PAYMENT INSTRUCTIONS
------------------------------------------------------------------
Option A: Stripe customer portal / payment link
  URL: {Stripe link generated per invoice or subscription}
  Accepts: Visa, Mastercard, American Express, SEPA Direct Debit,
           iDEAL, Bancontact, Apple Pay, Google Pay.

Option B: Bank transfer via Wise EUR account
  Account holder:  César Pereiro García
  IBAN:            {Wise EUR IBAN}
  BIC/SWIFT:       TRWIBEB1XXX (Wise Europe SA, Belgium)
  Reference:       Invoice FMM_FAIRNESS_EVAL_CLI-{YYYY}-{NNN}

Please include the invoice number in the payment reference.

------------------------------------------------------------------
NOTES
------------------------------------------------------------------
- Service governed by `legal/tos.md` and `legal/privacy.md`.
- Late payment after due date accrues statutory commercial
  interest per Spanish Ley 3/2004 transposing Directive 2011/7/EU.
- Queries: cesar.pereiro.garcia@gmail.com, subject
  "Invoice FMM_FAIRNESS_EVAL_CLI-{YYYY}-{NNN}".

------------------------------------------------------------------
SIGNATURE
------------------------------------------------------------------
Issued by:       César Pereiro García
Date:            {Issue date}
Signature:       {Digital signature or wet signature}
```

## VAT decision tree

1. Buyer established in Spain? → defer engagement until autónomo, OR use billing facilitator.
2. Buyer VAT-registered B2B in another EU Member State (validated via VIES)? → 0% VAT, EU reverse-charge under Art. 196 Dir. 2006/112/EC.
3. Buyer established outside the EU? → 0% VAT, out-of-scope supply.
4. Buyer is an EU consumer (B2C)? → DO NOT engage at current registration status; refer to a registered EU consultancy.
