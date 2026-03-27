#Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je plaćen po radnom satu. Koristite ugrađenu Python metodu input().
# Nakon toga izračunajte koliko je korisnik zaradio i ispišite na ekran. 
# Na kraju prepravite rješenje na način da ukupni iznos izračunavate u zasebnoj funkciji naziva total_euro.
#Primjer:
#Radni sati: 35 h
#eura/h: 8.5
#Ukupno: 297.5 eura

print("Unesi broj sati")
sati =float (input())
print("Unesi placu po satu")
placa = float(input())

ukupno = sati * placa

def total_euro(sati,placa):
    ukupno = sati * placa
    return ukupno

print("broj sati ", sati)
print("Kolio eura", placa)
print("Ukupno:", ukupno)
    