#Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja nekakvu ocjenu i nalazi se između 0.0 i 1.0.
#  Ispišite kojoj kategoriji pripada ocjena na temelju sljedećih uvjeta:
#>= 0.9 A
#>= 0.8 B
#>= 0.7 C
#>= 0.6 D
#< 0.6 F

print ("Unos Ocjene")
ocjena = float(input())

if ocjena < 0.0 or ocjena > 1.0:
    print("broj nije u intervalu")
elif ocjena >= 0.9:
        print("A")  
elif ocjena >= 0.8:
        print("B")
elif ocjena >= 0.7:
        print("C")
elif ocjena >= 0.6:
        print("D")
elif ocjena <= 0.6:
        print("F")

