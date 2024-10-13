"""
Description: This script takes a name as input and converts it to a bit sequence. 
Then it converts the bit sequence to PPM (Pulse Position Modulation) waveforms for M=2,4,8,16.
Functions:
    str_to_bits(name):
        Converts a string to its corresponding ASCII bit sequence.
        Args:
            name (str): The input string to be converted.
        Returns:
            str: The bit sequence representing the input string.
    ppm_symbols(bit_sequence, M):
        Converts a bit sequence to PPM symbols for a given M.
        Args:
            bit_sequence (str): The input bit sequence.
            M (int): The modulation order.
        Returns:
            list: A list of PPM symbols.
    find_graycode(symbol):
        Finds the Gray code equivalent of a given symbol.
        Args:
            symbol (str): The input symbol.
        Returns:
            int: The Gray code equivalent of the input symbol.
    ppm_graph(bit_sequence, M):
        Generates and plots the PPM waveform for a given bit sequence and modulation order M.
        Args:
            bit_sequence (str): The input bit sequence.
            M (int): The modulation order.
        Returns:
            None
Usage:
    The script prompts the user to enter a name, converts the name to a bit sequence, 
    and generates PPM waveforms for M=2, 4, 8, and 16.
"""
import numpy as np
import matplotlib.pyplot as plt


# metatrepei to string se ascii bits
def str_to_bits(name):
    bits=''
    for char in name:
        binary = format( ord(char) ,'08b' )
        bits += binary
    
    return bits



#metatrepei to bit_sequence se ppm M symbols
def ppm_symbols(bit_sequence, M):
    
    s_length= int(np.log2(M))
    
    remainder = len(bit_sequence) % s_length
    
    if remainder != 0:
         padding = s_length - remainder
         bit_sequence += '0' * padding

    symbols = [bit_sequence[i:i+s_length] for i in range(0, len(bit_sequence), s_length)]
   
    return symbols



def find_graycode(symbol):
    if symbol == '0' or symbol =='00' or symbol =='000' or symbol =='0000':
        return 0
    elif symbol == '1' or symbol == '01' or symbol == '001' or symbol =='0001':
        return 1
    elif symbol == '10' or symbol == '010' or symbol =='0010':
        return 2
    elif symbol == '11' or symbol == '011' or symbol =='0011':
        return 3
    elif symbol =='100' or symbol =='0100':
        return 4
    elif symbol =='101' or symbol =='0101' :
        return 5
    elif symbol =='110'  or symbol =='0110':
        return 6    
    elif symbol =='111'  or symbol =='0111':
        return 7
    elif symbol =='1000':
        return 8
    elif symbol =='1001':
        return 9
    elif symbol =='1010':
        return 10
    elif symbol =='1011':
        return 11
    elif symbol =='1100':
        return 12
    elif symbol =='1101':
        return 13
    elif symbol =='1110':
        return 14
    elif symbol =='1111':
        return 15
    
    
#make the ppm waveform of M
def ppm_graph(bit_sequence,M):
    symbols = ppm_symbols(bit_sequence, M)
    
    #print the symbol list
    print(f"bit list for M={M} is:",symbols)
    
    #print the number of symbols in the list
    print(f"number of symbols for M={M} is:",len(symbols))
    
    Ts = 1e-9
    t = np.arange(0, len(symbols) * Ts,Ts/1000)  
    x = np.zeros(t.size)



    for  i, symbol in enumerate(symbols):
        k = find_graycode(symbol)
            
        pulse_duration = Ts / M
        on_time = np.where( (t>= i * Ts + k*pulse_duration) & (t< i * Ts + (k+1)*pulse_duration ) )
        x[on_time] = 1
        
        

        
        

    plt.close('all')
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, )
    plt.title(f" {M}-ppm")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.show()
 
   


#user input
name = input("enter name:")

#turn user input into bits
bit_sequence = str_to_bits(name)

#print the bit_sequence
print(f"bit sequence for {name} is:",bit_sequence)

#print the number of bits in the sequence
print("bits of name:",len(bit_sequence))

#ppm waveforms
ppm_graph(bit_sequence, 2)
ppm_graph(bit_sequence, 4)
ppm_graph(bit_sequence, 8)
ppm_graph(bit_sequence, 16)





