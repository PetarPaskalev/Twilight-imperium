"use client";

/**
 * OAuth Callback Page
 * 
 * Handles OAuth redirects from Google sign-in.
 * This page processes the auth code and redirects to the main chat page.
 */

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { supabase } from '../../../lib/supabase';

export default function AuthCallback() {
  const router = useRouter();

  useEffect(() => {
    // Handle the OAuth callback with retry logic
    const handleCallback = async (retryCount = 0) => {
      const MAX_RETRIES = 3;
      const RETRY_DELAY = 1000; // 1 second
      
      try {
        // Get the code from URL parameters
        const hashParams = new URLSearchParams(window.location.hash.substring(1));
        const code = hashParams.get('code') || new URLSearchParams(window.location.search).get('code');
        
        if (code) {
          // Exchange the code for a session (PKCE flow)
          const { data, error } = await supabase.auth.exchangeCodeForSession(code);
          
          if (error) {
            console.error('Error exchanging code for session:', error);
            
            // Retry on network errors or temporary failures
            if (retryCount < MAX_RETRIES && (
              error.message?.includes('network') || 
              error.message?.includes('timeout') ||
              error.message?.includes('fetch')
            )) {
              console.log(`Retrying auth callback (attempt ${retryCount + 1}/${MAX_RETRIES})...`);
              setTimeout(() => handleCallback(retryCount + 1), RETRY_DELAY * (retryCount + 1));
              return;
            }
            
            router.push('/?error=auth_failed');
            return;
          }
          
          // Success - wait a moment for session to be fully established, then redirect
          setTimeout(() => {
            router.push('/');
          }, 500);
        } else {
          // Try to get existing session (for implicit flow or if already authenticated)
          const { data: { session }, error } = await supabase.auth.getSession();
          
          if (error || !session) {
            console.error('No session found after OAuth callback:', error);
            
            // Retry on network errors
            if (retryCount < MAX_RETRIES && (
              error?.message?.includes('network') || 
              error?.message?.includes('timeout') ||
              error?.message?.includes('fetch')
            )) {
              console.log(`Retrying session fetch (attempt ${retryCount + 1}/${MAX_RETRIES})...`);
              setTimeout(() => handleCallback(retryCount + 1), RETRY_DELAY * (retryCount + 1));
              return;
            }
            
            router.push('/?error=auth_failed');
            return;
          }
          
          // Session exists - wait a moment, then redirect to home
          setTimeout(() => {
            router.push('/');
          }, 500);
        }
      } catch (error) {
        console.error('Unexpected error during auth callback:', error);
        
        // Retry on unexpected errors (might be network issues)
        if (retryCount < MAX_RETRIES) {
          console.log(`Retrying after unexpected error (attempt ${retryCount + 1}/${MAX_RETRIES})...`);
          setTimeout(() => handleCallback(retryCount + 1), RETRY_DELAY * (retryCount + 1));
          return;
        }
        
        router.push('/?error=auth_failed');
      }
    };

    handleCallback();
  }, [router]);

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        background: '#0b0f14',
        color: '#e5e7eb',
      }}
    >
      <div style={{ textAlign: 'center' }}>
        <div
          style={{
            width: '50px',
            height: '50px',
            border: '4px solid #333',
            borderTop: '4px solid #2563eb',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 1rem',
          }}
        />
        <p style={{ fontSize: '1.1rem' }}>Completing sign in...</p>
        <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    </div>
  );
}

