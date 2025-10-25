"use client";

/**
 * UserProfile - Displays user information and usage statistics inline in header
 * 
 * Shows user's email, tier, daily message usage, and logout button.
 */

import { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type UsageData = {
  user_id: string;
  email: string;
  tier: string;
  usage: {
    used: number;
    limit: number;
    remaining: number;
  };
};

export default function UserProfile() {
  const { user, userProfile, session, signOut } = useAuth();
  const [usage, setUsage] = useState<UsageData | null>(null);
  const [loading, setLoading] = useState(false);

  // Fetch user usage data from backend
  useEffect(() => {
    if (user && session) {
      fetchUsage();
    }
  }, [user, session]);

  const fetchUsage = async () => {
    if (!user || !session) return;
    
    setLoading(true);
    try {
      const token = session.access_token;
      
      if (!token) return;

      const res = await fetch(`${API_URL}/me`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (res.ok) {
        const data = await res.json();
        setUsage(data);
      }
    } catch (error) {
      console.error('Error fetching usage:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!user || !userProfile) return null;

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '1rem',
      }}
    >
      {/* User Info */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div>
          <div
            style={{
              fontSize: '0.85rem',
              color: '#e5e7eb',
              marginBottom: '0.125rem',
            }}
          >
            {userProfile.email}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div
              style={{
                display: 'inline-block',
                padding: '0.125rem 0.5rem',
                borderRadius: '4px',
                backgroundColor: userProfile.tier === 'paid' ? '#0070f3' : '#555',
                color: '#fff',
                fontSize: '0.7rem',
                fontWeight: 'bold',
                textTransform: 'uppercase',
              }}
            >
              {userProfile.tier}
            </div>
            {/* Usage indicator */}
            {usage && (
              <div
                style={{
                  fontSize: '0.75rem',
                  color: usage.usage.remaining < 5 ? '#ff9944' : '#9aa0a6',
                }}
              >
                {usage.usage.used}/{usage.usage.limit} msgs
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Sign Out Button */}
      <button
        onClick={signOut}
        style={{
          padding: '0.5rem 1rem',
          borderRadius: '6px',
          border: '1px solid #374151',
          backgroundColor: '#1f2937',
          color: '#e5e7eb',
          fontSize: '0.875rem',
          cursor: 'pointer',
          transition: 'background-color 0.2s',
          whiteSpace: 'nowrap',
        }}
        onMouseOver={(e) => (e.currentTarget.style.backgroundColor = '#374151')}
        onMouseOut={(e) => (e.currentTarget.style.backgroundColor = '#1f2937')}
      >
        Sign Out
      </button>
    </div>
  );
}

