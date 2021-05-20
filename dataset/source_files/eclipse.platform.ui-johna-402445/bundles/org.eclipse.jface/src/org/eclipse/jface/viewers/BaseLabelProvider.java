/*******************************************************************************
 * Copyright (c) 2006, 2013 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *     Hendrik Still <hendrik.still@gammas.de> - bug 412273
 *******************************************************************************/

package org.eclipse.jface.viewers;

import org.eclipse.core.commands.common.EventManager;
import org.eclipse.jface.util.SafeRunnable;

/**
 * BaseLabelProvider is a default concrete implementation of
 * {@link IBaseLabelProvider} 
 * @param <E> Type of an element of the model
 * 
 * @since 3.3
 * 
 */
public class BaseLabelProvider<E> extends EventManager implements IBaseLabelProvider<E> {
	
	/* (non-Javadoc)
     * Method declared on IBaseLabelProvider.
     */
    public void addListener(ILabelProviderListener<E> listener) {
        addListenerObject(listener);
    }

    /**
     * The <code>BaseLabelProvider</code> implementation of this 
     * <code>IBaseLabelProvider</code> method clears its internal listener list.
     * Subclasses may extend but should call the super implementation.
     */
    public void dispose() {
    	clearListeners();
    }
    
    /**
     * The <code>BaseLabelProvider</code> implementation of this 
     * <code>IBaseLabelProvider</code> method returns <code>true</code>. Subclasses may 
     * override.
     */
    public boolean isLabelProperty(E element, String property) {
        return true;
    }

    
    /* (non-Javadoc)
     * @see org.eclipse.jface.viewers.IBaseLabelProvider#removeListener(org.eclipse.jface.viewers.ILabelProviderListener)
     */
    public void removeListener(ILabelProviderListener<E> listener) {
        removeListenerObject(listener);
    }
    
    /**
	 * Fires a label provider changed event to all registered listeners Only
	 * listeners registered at the time this method is called are notified.
	 * 
	 * @param event
	 *            a label provider changed event
	 * 
	 * @see ILabelProviderListener#labelProviderChanged
	 */
	protected void fireLabelProviderChanged(final LabelProviderChangedEvent<E> event) {
		Object[] listeners = getListeners();
		for (int i = 0; i < listeners.length; ++i) {
			@SuppressWarnings("unchecked")
			final ILabelProviderListener<E> l = (ILabelProviderListener<E>) listeners[i];
			SafeRunnable.run(new SafeRunnable() {
				public void run() {
					l.labelProviderChanged(event);
				}
			});

		}
	}
}
