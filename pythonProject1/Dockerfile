# Χρησιμοποιήστε ένα επίσημο image του Python ως βάση
FROM python:3.9

# Ορίστε το working directory μέσα στο container
WORKDIR /usr/src/app

# Αντιγράψτε το requirements.txt στο container
COPY requirements.txt ./

# Εγκαταστήστε τις εξαρτήσεις της εφαρμογής
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγράψτε όλα τα αρχεία της εφαρμογής
COPY . .

# Εκθέστε τη θύρα στην οποία τρέχει η εφαρμογή (π.χ., 5000)
EXPOSE 5000

# Ορίστε την εντολή που θα εκτελείται κατά την εκκίνηση του container
CMD ["python", "main.py"]

