"use client";

/**
 * AuthModal - Modal component for login and signup
 * 
 * Provides a clean modal interface for user authentication with email/password
 * and social login options (Google, GitHub).
 */

import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

type AuthModalProps = {
  isOpen: boolean;
  onClose: () => void;
};

export default function AuthModal({ isOpen, onClose }: AuthModalProps) {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { signIn, signUp, signInWithProvider } = useAuth();

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isLogin) {
        // Login
        const { error } = await signIn(email, password);
        if (error) {
          setError(error.message);
        } else {
          onClose();
        }
      } else {
        // Signup
        const { error } = await signUp(email, password, fullName);
        if (error) {
          setError(error.message);
        } else {
          onClose();
        }
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleSocialLogin = async (provider: 'google' | 'github') => {
    setError('');
    setLoading(true);
    try {
      await signInWithProvider(provider);
    } catch (err: any) {
      setError(err.message || 'Social login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#1e1e1e',
          padding: '2rem',
          borderRadius: '12px',
          width: '90%',
          maxWidth: '400px',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ marginBottom: '1.5rem', textAlign: 'center' }}>
          <h2 style={{ margin: 0, fontSize: '1.5rem', color: '#fff' }}>
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </h2>
          <p style={{ margin: '0.5rem 0 0', color: '#888', fontSize: '0.9rem' }}>
            {isLogin ? 'Sign in to continue chatting' : 'Sign up to get started'}
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div
            style={{
              backgroundColor: '#ff4444',
              color: '#fff',
              padding: '0.75rem',
              borderRadius: '6px',
              marginBottom: '1rem',
              fontSize: '0.9rem',
            }}
          >
            {error}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit}>
          {!isLogin && (
            <div style={{ marginBottom: '1rem' }}>
              <label
                style={{
                  display: 'block',
                  marginBottom: '0.5rem',
                  color: '#ccc',
                  fontSize: '0.9rem',
                }}
              >
                Full Name
              </label>
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="John Doe"
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  borderRadius: '6px',
                  border: '1px solid #444',
                  backgroundColor: '#2a2a2a',
                  color: '#fff',
                  fontSize: '1rem',
                  boxSizing: 'border-box',
                }}
              />
            </div>
          )}

          <div style={{ marginBottom: '1rem' }}>
            <label
              style={{
                display: 'block',
                marginBottom: '0.5rem',
                color: '#ccc',
                fontSize: '0.9rem',
              }}
            >
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="you@example.com"
              style={{
                width: '100%',
                padding: '0.75rem',
                borderRadius: '6px',
                border: '1px solid #444',
                backgroundColor: '#2a2a2a',
                color: '#fff',
                fontSize: '1rem',
                boxSizing: 'border-box',
              }}
            />
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label
              style={{
                display: 'block',
                marginBottom: '0.5rem',
                color: '#ccc',
                fontSize: '0.9rem',
              }}
            >
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              placeholder="••••••••"
              minLength={6}
              style={{
                width: '100%',
                padding: '0.75rem',
                borderRadius: '6px',
                border: '1px solid #444',
                backgroundColor: '#2a2a2a',
                color: '#fff',
                fontSize: '1rem',
                boxSizing: 'border-box',
              }}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              padding: '0.875rem',
              borderRadius: '6px',
              border: 'none',
              backgroundColor: loading ? '#555' : '#0070f3',
              color: '#fff',
              fontSize: '1rem',
              fontWeight: 'bold',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'background-color 0.2s',
            }}
          >
            {loading ? 'Processing...' : isLogin ? 'Sign In' : 'Sign Up'}
          </button>
        </form>

        {/* Social Login Buttons */}
        <div style={{ margin: '1.5rem 0' }}>
          <div
            style={{
              textAlign: 'center',
              position: 'relative',
              marginBottom: '1rem',
            }}
          >
            <span
              style={{
                backgroundColor: '#1e1e1e',
                padding: '0 0.5rem',
                color: '#888',
                fontSize: '0.9rem',
              }}
            >
              or continue with
            </span>
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: 0,
                right: 0,
                height: '1px',
                backgroundColor: '#444',
                zIndex: -1,
              }}
            />
          </div>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              onClick={() => handleSocialLogin('google')}
              disabled={loading}
              style={{
                flex: 1,
                padding: '0.75rem',
                borderRadius: '6px',
                border: '1px solid #444',
                backgroundColor: '#2a2a2a',
                color: '#fff',
                fontSize: '0.9rem',
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'background-color 0.2s',
              }}
            >
              Google
            </button>
            <button
              type="button"
              onClick={() => handleSocialLogin('github')}
              disabled={loading}
              style={{
                flex: 1,
                padding: '0.75rem',
                borderRadius: '6px',
                border: '1px solid #444',
                backgroundColor: '#2a2a2a',
                color: '#fff',
                fontSize: '0.9rem',
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'background-color 0.2s',
              }}
            >
              GitHub
            </button>
          </div>
        </div>

        {/* Toggle Login/Signup */}
        <div style={{ textAlign: 'center', color: '#888', fontSize: '0.9rem' }}>
          {isLogin ? "Don't have an account? " : 'Already have an account? '}
          <button
            type="button"
            onClick={() => {
              setIsLogin(!isLogin);
              setError('');
            }}
            style={{
              background: 'none',
              border: 'none',
              color: '#0070f3',
              cursor: 'pointer',
              textDecoration: 'underline',
              fontSize: '0.9rem',
            }}
          >
            {isLogin ? 'Sign Up' : 'Sign In'}
          </button>
        </div>
      </div>
    </div>
  );
}

