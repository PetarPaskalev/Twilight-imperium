"use client";

/**
 * AuthContext - Provides authentication state and methods throughout the app
 * 
 * This context manages user authentication state, login, signup, and logout functionality.
 * It also handles automatic profile creation and session persistence.
 */

import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { User, Session, AuthError } from '@supabase/supabase-js';
import { supabase, UserProfile } from '../lib/supabase';

type AuthContextType = {
  user: User | null;
  session: Session | null;
  userProfile: UserProfile | null;
  loading: boolean;
  signUp: (email: string, password: string, fullName?: string) => Promise<{ error: AuthError | null }>;
  signIn: (email: string, password: string) => Promise<{ error: AuthError | null }>;
  signOut: () => Promise<void>;
  signInWithProvider: (provider: 'google') => Promise<void>;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);

  // Fetch user profile from database, create if it doesn't exist (for OAuth users)
  const fetchUserProfile = async (userId: string, userEmail?: string, userFullName?: string) => {
    try {
      const { data, error } = await supabase
        .from('user_profiles')
        .select('*')
        .eq('id', userId)
        .single();

      if (error) {
        // If profile doesn't exist (PGRST116 = no rows returned), create it
        // This happens for first-time OAuth users
        if (error.code === 'PGRST116' && userEmail) {
          console.log('Profile not found, creating new profile for OAuth user');
          await createUserProfile(userId, userEmail, userFullName);
          // Fetch the newly created profile
          const { data: newData } = await supabase
            .from('user_profiles')
            .select('*')
            .eq('id', userId)
            .single();
          if (newData) {
            setUserProfile(newData);
          }
        } else {
          throw error;
        }
      } else {
        setUserProfile(data);
      }
    } catch (error) {
      console.error('Error fetching user profile:', error);
      setUserProfile(null);
    }
  };

  // Create user profile if it doesn't exist
  const createUserProfile = async (userId: string, email: string, fullName?: string) => {
    try {
      const { data, error } = await supabase
        .from('user_profiles')
        .insert({
          id: userId,
          email: email,
          full_name: fullName || null,
          tier: 'free',
        })
        .select()
        .single();

      if (error) throw error;
      setUserProfile(data);
    } catch (error) {
      console.error('Error creating user profile:', error);
    }
  };

  // Initialize auth state on mount
  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      if (session?.user) {
        const userEmail = session.user.email || undefined;
        const userFullName = session.user.user_metadata?.full_name || 
                           session.user.user_metadata?.name || 
                           undefined;
        fetchUserProfile(session.user.id, userEmail, userFullName);
      }
      setLoading(false);
    });

    // Listen for auth state changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (_event, session) => {
      setSession(session);
      setUser(session?.user ?? null);
      
      if (session?.user) {
        // Get user info for profile creation (important for OAuth users)
        const userEmail = session.user.email || undefined;
        const userFullName = session.user.user_metadata?.full_name || 
                           session.user.user_metadata?.name || 
                           undefined;
        await fetchUserProfile(session.user.id, userEmail, userFullName);
      } else {
        setUserProfile(null);
      }
      
      setLoading(false);
    });

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  // Sign up with email and password
  const signUp = async (email: string, password: string, fullName?: string) => {
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
      });

      if (error) return { error };

      // Create user profile after successful signup
      if (data.user) {
        await createUserProfile(data.user.id, email, fullName);
      }

      return { error: null };
    } catch (error: any) {
      return { error };
    }
  };

  // Sign in with email and password
  const signIn = async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      return { error };
    } catch (error: any) {
      return { error };
    }
  };

  // Sign out
  const signOut = async () => {
    await supabase.auth.signOut();
    setUser(null);
    setSession(null);
    setUserProfile(null);
  };

  // Sign in with Google OAuth
  const signInWithProvider = async (provider: 'google') => {
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider,
        options: {
          redirectTo: `${window.location.origin}/auth/callback`,
        },
      });

      if (error) throw error;
    } catch (error) {
      console.error('Error signing in with Google:', error);
    }
  };

  const value = {
    user,
    session,
    userProfile,
    loading,
    signUp,
    signIn,
    signOut,
    signInWithProvider,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// Custom hook to use auth context
export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

