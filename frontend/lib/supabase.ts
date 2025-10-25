/**
 * Supabase client configuration
 * 
 * This file creates and exports a Supabase client instance
 * used for authentication and database operations.
 */

import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

// Create a single supabase client for interacting with your database
export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
    // Store session in localStorage
    storage: typeof window !== 'undefined' ? window.localStorage : undefined,
  },
});

// Type definitions for our database tables
export type UserProfile = {
  id: string;
  email: string | null;
  full_name: string | null;
  avatar_url: string | null;
  tier: 'free' | 'paid';
  created_at: string;
  updated_at: string;
};

export type UserUsage = {
  id: string;
  user_id: string;
  date: string;
  message_count: number;
  created_at: string;
  updated_at: string;
};

