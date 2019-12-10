/*
 *  tsmtv_helpers.h
 *
 
 Includes and helper functions for MEX functions of TomoSegMemTV on Windows.
 
 *
 *  Created by Antonio Martinez on 02/04/14.
 *  Copyright 2014 Universidad Almeria / Max Planck Insititute of Biochemisty. All rights reserved.
 *
 */
/////////////////// INCLUDES ///////////////////////////

#include <windows.h>
#include <malloc.h>    
#include <stdio.h>
#include <mex.h>
#include <math.h>
#include <pthread.h>
//#include <unistd.h>
#include <float.h>
#include <string.h>

////////////////// DECLARATION ////////////////////////

#define INTER_FACTOR .71
#define CACHE_USED .33
#define BUFFER_SIZE 1024
#define BYTE_PER_KBYTE 1024
#define ERROR_FILE_READ -1
#define ERROR_CACHE_NOT_FOUND -2
#define ERROR_MEM_NOT_FOUND -3
#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)

#define SQR(x)      ((x)*(x)) 

typedef BOOL (WINAPI *LPFN_GLPI)(
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, 
    PDWORD);

long long int get_cache_size(); 
long long int get_mem_size(); 
int get_num_processors();
DWORD CountSetBits(ULONG_PTR bitMask);

////////////////// IMPLEMENTATION //////////////////////

// Get the number of logical processors, return , 0 if fails
int get_num_processors()
{
	LPFN_GLPI glpi;
    BOOL done = FALSE;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = NULL;
    DWORD returnLength = 0;
    DWORD logicalProcessorCount = 0;
	DWORD byteOffset = 0;

	glpi = (LPFN_GLPI) GetProcAddress(
                            GetModuleHandle(TEXT("kernel32")),
                            "GetLogicalProcessorInformation");
    glpi = (LPFN_GLPI) GetProcAddress(
                            GetModuleHandle(TEXT("kernel32")),
                            "GetLogicalProcessorInformation");
    if (NULL == glpi) 
    {
        return (-1);
    }

    while (!done)
    {
        DWORD rc = glpi(buffer, &returnLength);

        if (FALSE == rc) 
        {
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) 
            {
                if (buffer) 
                    free(buffer);

                buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(
                        returnLength);

                if (NULL == buffer) 
                {
                    return (-2);
                }
            } 
            else 
            {
                return (-3);
            }
        } 
        else
        {
            done = TRUE;
        }
    }

    ptr = buffer;

    while (byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= returnLength) 
    {
        switch (ptr->Relationship) 
        {
        case RelationProcessorCore:
            // A hyperthreaded core supplies more than one logical processor.
            logicalProcessorCount += CountSetBits(ptr->ProcessorMask);
            break;
        }
        byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;
    }

	return (int)logicalProcessorCount;
}

// Get the size in bytes of the RAM memory, return , 0 if fails
long long int get_mem_size()
{
	MEMORYSTATUSEX statex; 
	
	statex.dwLength = sizeof(statex); // I misunderstand that

	GlobalMemoryStatusEx( &statex );
	
	return ((long long int)statex.ullTotalPhys); 
}

long long int get_cache_size()
{
	LPFN_GLPI glpi;
    BOOL done = FALSE;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = NULL;
    DWORD returnLength = 0;
	DWORD byteOffset = 0;
	PCACHE_DESCRIPTOR Cache;
	long long int cache_size = 0;

	glpi = (LPFN_GLPI) GetProcAddress(
                            GetModuleHandle(TEXT("kernel32")),
                            "GetLogicalProcessorInformation");
    glpi = (LPFN_GLPI) GetProcAddress(
                            GetModuleHandle(TEXT("kernel32")),
                            "GetLogicalProcessorInformation");
    if (NULL == glpi) 
    {
        return (-1);
    }

    while (!done)
    {
        DWORD rc = glpi(buffer, &returnLength);

        if (FALSE == rc) 
        {
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) 
            {
                if (buffer) 
                    free(buffer);

                buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(
                        returnLength);

                if (NULL == buffer) 
                {
                    return (-2);
                }
            } 
            else 
            {
                return (-3);
            }
        } 
        else
        {
            done = TRUE;
        }
    }

    ptr = buffer;

    while (byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= returnLength) 
    {
        switch (ptr->Relationship) 
        {
        case RelationCache:
            // Cache data is in ptr->Cache, one CACHE_DESCRIPTOR structure for each cache.
			// Return the maximum cache size found
            Cache = &ptr->Cache;
            if (Cache->Size > cache_size)
            {
				cache_size = (long long int)Cache->Size;
            }
            break;
        }
        byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;
    }

	return cache_size;
}

// Helper function to count set bits in the processor mask.
DWORD CountSetBits(ULONG_PTR bitMask)
{
    DWORD LSHIFT = sizeof(ULONG_PTR)*8 - 1;
    DWORD bitSetCount = 0;
    ULONG_PTR bitTest = (ULONG_PTR)1 << LSHIFT;    
    DWORD i;
    
    for (i = 0; i <= LSHIFT; ++i)
    {
        bitSetCount += ((bitMask & bitTest)?1:0);
        bitTest/=2;
    }

    return bitSetCount;
}
