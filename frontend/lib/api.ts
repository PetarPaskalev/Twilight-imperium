/**
 * API Configuration and Utilities
 * 
 * Centralized API URL configuration with smart fallbacks
 * and retry logic for production reliability.
 */

// Determine if we're in production
const isProduction = typeof window !== 'undefined' && 
  (window.location.hostname !== 'localhost' && 
   window.location.hostname !== '127.0.0.1');

// Production API URL (Render backend)
const PRODUCTION_API_URL = 'https://twilight-imperium.onrender.com';

// Development API URL (local backend)
const DEVELOPMENT_API_URL = 'http://localhost:8000';

// Get API URL with smart fallback
export const getApiUrl = (): string => {
  // First, check environment variable (set in Vercel)
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  
  // Fallback: use production URL if we're not on localhost
  if (isProduction) {
    return PRODUCTION_API_URL;
  }
  
  // Development fallback
  return DEVELOPMENT_API_URL;
};

// Export the API URL constant
export const API_URL = getApiUrl();

// Log API URL in development for debugging
if (typeof window !== 'undefined' && !isProduction) {
  console.log('🔧 API URL:', API_URL);
  console.log('🔧 Environment variable:', process.env.NEXT_PUBLIC_API_URL || 'not set');
}

/**
 * Fetch with retry logic for better reliability
 */
export async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  maxRetries: number = 3,
  retryDelay: number = 1000
): Promise<Response> {
  let lastError: Error | null = null;
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);
      
      // If successful, return immediately
      if (response.ok) {
        return response;
      }
      
      // Don't retry on client errors (4xx) except 429 (rate limit)
      if (response.status >= 400 && response.status < 500 && response.status !== 429) {
        return response;
      }
      
      // For server errors (5xx) or rate limits, throw to trigger retry
      if (response.status >= 500 || response.status === 429) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return response;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      
      // Don't retry on the last attempt
      if (attempt < maxRetries - 1) {
        // Exponential backoff: wait longer between retries
        const delay = retryDelay * Math.pow(2, attempt);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  
  // If all retries failed, throw the last error
  throw lastError || new Error('Request failed after retries');
}

