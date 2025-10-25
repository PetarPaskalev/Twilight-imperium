"use client";

/**
 * UserProfile - Displays user information and usage statistics
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
        position: 'absolute',
        top: '1rem',
        right: '1rem',
        backgroundColor: '#1e1e1e',
        padding: '1rem',
        borderRadius: '8px',
        border: '1px solid #333',
        minWidth: '200px',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
      }}
    >
      {/* User Info */}
      <div style={{ marginBottom: '0.75rem' }}>
        <div
          style={{
            fontSize: '0.9rem',
            color: '#ccc',
            marginBottom: '0.25rem',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {userProfile.email}
        </div>
        <div
          style={{
            display: 'inline-block',
            padding: '0.25rem 0.5rem',
            borderRadius: '4px',
            backgroundColor: userProfile.tier === 'paid' ? '#0070f3' : '#555',
            color: '#fff',
            fontSize: '0.75rem',
            fontWeight: 'bold',
            textTransform: 'uppercase',
          }}
        >
          {userProfile.tier}
        </div>
      </div>

      {/* Usage Stats */}
      {loading ? (
        <div style={{ color: '#888', fontSize: '0.85rem', marginBottom: '0.75rem' }}>
          Loading usage...
        </div>
      ) : usage ? (
        <div
          style={{
            marginBottom: '0.75rem',
            padding: '0.5rem',
            backgroundColor: '#2a2a2a',
            borderRadius: '6px',
          }}
        >
          <div
            style={{
              fontSize: '0.8rem',
              color: '#888',
              marginBottom: '0.25rem',
            }}
          >
            Today's Messages
          </div>
          <div style={{ fontSize: '1.25rem', color: '#fff', fontWeight: 'bold' }}>
            {usage.usage.used} / {usage.usage.limit}
          </div>
          <div
            style={{
              marginTop: '0.5rem',
              height: '4px',
              backgroundColor: '#444',
              borderRadius: '2px',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${(usage.usage.used / usage.usage.limit) * 100}%`,
                backgroundColor:
                  usage.usage.remaining === 0
                    ? '#ff4444'
                    : usage.usage.remaining < 5
                    ? '#ff9944'
                    : '#44ff44',
                transition: 'width 0.3s',
              }}
            />
          </div>
          <div style={{ fontSize: '0.75rem', color: '#888', marginTop: '0.25rem' }}>
            {usage.usage.remaining} remaining
          </div>
        </div>
      ) : null}

      {/* Logout Button */}
      <button
        onClick={signOut}
        style={{
          width: '100%',
          padding: '0.5rem',
          borderRadius: '6px',
          border: '1px solid #444',
          backgroundColor: '#2a2a2a',
          color: '#fff',
          fontSize: '0.85rem',
          cursor: 'pointer',
          transition: 'background-color 0.2s',
        }}
        onMouseOver={(e) => (e.currentTarget.style.backgroundColor = '#333')}
        onMouseOut={(e) => (e.currentTarget.style.backgroundColor = '#2a2a2a')}
      >
        Sign Out
      </button>
    </div>
  );
}

